# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from PIL import Image
from itertools import product
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import julius
from scipy.signal import fftconvolve, butter, sosfiltfilt
from scipy.io import wavfile

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)


def to_tensor(v):
    import numpy as np

    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def compute_speech_file(split, speech_id):
    split_mapping = {'train': 'train-clean-360', 'val': 'dev-clean', 'val-seen': 'dev-clean', 'test': 'test-clean',
                     'test-seen': 'test-clean'}
    book_id, ch_id, sec_id = speech_id.split('-')
    path = os.path.join(f'data/sounds/speech/LibriSpeech', split_mapping[split], book_id, ch_id, speech_id + '.flac')
    return path


def compute_relative_pose(v1, v2):
    # v2's pose in v1's coordinate frame
    d = np.sqrt((v1['location'][0] - v2['location'][0]) ** 2 + (v1['location'][2] - v2['location'][2]) ** 2)
    theta = np.arctan2(v2['location'][2] - v1['location'][2], v2['location'][0] - v1['location'][0])
    # facing direction is 0 and goes anti clockwise
    relative_rotation = ((np.deg2rad(270) - theta) - v1['rotation']) % (np.pi * 2)
    return [d, relative_rotation]


class SoundSpacesNVASDataset(Dataset):
    def __init__(self, split, args):
        super().__init__()
        self.split = split
        self.args = args
        self.image_res = (256, 256)
        self.audio_len = args.audio_len

        data_dir = f'{args.dataset_dir}/{split}'
        scenes = os.listdir(data_dir)
        self.metadata_scene_list = []
        for scene in scenes:
            metadata_file = os.path.join(data_dir, scene, args.metadata_file)
            if not os.path.exists(metadata_file):
                continue

            with open(metadata_file, 'r') as fo:
                metadata_list = json.load(fo)
                self.metadata_scene_list += [(scene, metadata) for metadata in metadata_list]

        print(f'Number of clip is {len(self.metadata_scene_list)} for {self.split.upper()}')

    def __len__(self):
        return len(self.metadata_scene_list)

    def __getitem__(self, item):
        scene, metadata = self.metadata_scene_list[item]
        viewpoints = metadata['viewpoints']
        speakers = metadata['speakers']

        if self.args.eval_difficult:
            max_distance = 0
            for a, b in product(viewpoints, viewpoints):
                distance = np.sqrt(np.sum(np.array(a['location']) - np.array(b['location']))**2)
                if distance > max_distance:
                    max_distance = distance
                    v1 = a
                    v2 = b
        else:
            v1, v2 = np.random.permutation(viewpoints)[:2]

        audios = []
        if self.args.one_speaker:
            active_spk_indices = [np.random.choice([spk['id'] for spk in speakers if spk['speech'] is not None])]
        else:
            active_spk_indices = [spk['id'] for spk in speakers if spk['speech'] is not None]
        active_speeches = dict()
        for spk_id in active_spk_indices:
            if self.args.use_same_speech:
                speech_file = 'data/sounds/speech/LibriSpeech/train-clean-360/100/121669/100-121669-0004.flac'
            else:
                speech_file = compute_speech_file(self.split, speakers[int(spk_id)]['speech'])
            sr, speech = wavfile.read(speech_file.replace('LibriSpeech', 'LibriSpeech-wav'))
            if self.args.highpass_filter:
                speech = butter_bandpass_filter(speech, 80, sr//2 - 1, sr, order=5)
            if sr != self.args.sample_rate:
                print('Resample audio')
                speech = julius.resample_frac(torch.tensor(speech), sr, self.args.sample_rate).numpy()

            if self.args.randomize_speech_amplitude and self.split == 'train':
                if np.random.random() < 0.5:
                    scale = np.random.random() * 0.9 + 0.1
                else:
                    scale = np.random.random() * 9 + 1
                speech *= scale

            active_speeches[spk_id] = speech

        for i, viewpoint in enumerate([v1, v2]):
            received_audio = []
            rir_dict = viewpoint['mono_rir'] if ((self.args.input_mono and i == 0) or (self.args.output_mono and i == 1)
                                                 ) else viewpoint['rir']
            for spk_id, rir_file in rir_dict.items():
                if int(spk_id) not in active_spk_indices:
                    continue

                sr, rir = wavfile.read(rir_file)
                rir = rir.T
                if len(rir.shape) == 1:
                    rir = np.expand_dims(rir, -1)
                if (self.args.input_mono and i == 0) or (self.args.output_mono and i == 1):
                    rir = np.stack([rir, rir])[..., 0]
                if sr != self.args.sample_rate:
                    print('Resample audio')
                    rir = julius.resample_frac(torch.tensor(rir), sr, self.args.sample_rate).numpy()
                if self.args.remove_delay and not np.allclose(rir, 0):
                    rir = rir[:, min([np.min(np.nonzero(rir[0])), np.min(np.nonzero(rir[1]))]):]

                received_audio.append(np.array([fftconvolve(active_speeches[int(spk_id)], rir_channel)
                                                for rir_channel in rir]))
            audio_sum = np.zeros([2, max([audio.shape[1] for audio in received_audio])])
            for audio in received_audio:
                audio_sum[:, :audio.shape[1]] += audio
            audios.append(audio_sum)

        tgt_pose = compute_relative_pose(v1, v2) if not self.args.fix_tgt_pose else [1, 1]
        tgt_rotation = [(v2['rotation'] - v1['rotation']) % (np.pi * 2)]

        sample = dict()
        a1, a2 = self.process_audio(audios[0]), self.process_audio(audios[1])

        if self.args.use_clean_speech:
            assert self.args.one_speaker
            a1 = self.process_audio(np.stack([active_speeches[active_spk_indices[0]],
                                              active_speeches[active_spk_indices[0]]], axis=0))
        if self.args.use_clean_as_tgt:
            a2 = self.process_audio(np.stack([active_speeches[active_spk_indices[0]],
                                              active_speeches[active_spk_indices[0]]], axis=0))

        speech_len = active_speeches[active_spk_indices[0]].shape[0]
        if speech_len > self.audio_len and not self.args.use_same_speech \
                and not self.args.no_prev_reverb and self.split == 'train':
            random_start = np.random.randint(speech_len - self.audio_len)
            sample['src_wav'] = to_tensor(a1[:, random_start: random_start + self.audio_len])
            sample['tgt_wav'] = to_tensor(a2[:, random_start: random_start + self.audio_len])
        else:
            sample['src_wav'] = to_tensor(a1[:, :self.audio_len])
            sample['tgt_wav'] = to_tensor(a2[:, :self.audio_len])

        if self.args.normalize_input:
            peak_value = sample['src_wav'].abs().max()
            if peak_value != 0:
                sample['src_wav'] /= peak_value
                sample['tgt_wav'] /= peak_value

        if self.args.encode_sincos:
            tgt_pose = [tgt_pose[0], np.sin(tgt_pose[1]), np.cos(tgt_pose[1])]
            tgt_rotation = [np.sin(tgt_rotation[0]), np.cos(tgt_rotation[0])]

        sample['src_rgb'] = self.process_rgb(np.array(Image.open(v1['image'])).astype(np.float32))
        sample['tgt_rgb'] = self.process_rgb(np.array(Image.open(v2['image'])).astype(np.float32))
        if self.args.use_tgt_rgb_pano:
            image = np.array(Image.open(v1['image'][:-4] + '_pano.png')).astype(np.float32)
            image = to_tensor(image).permute(2, 0, 1) / 255.0
            image = torchvision.transforms.Resize((192, 576))(image)
            if self.split == 'train':
                width_shift = np.random.randint(0, image.shape[-1])
                sample['tgt_rgb_pano'] = torch.roll(image, width_shift, dims=-1)
            else:
                sample['tgt_rgb_pano'] = image
        if self.args.use_tgt_bboxes_as_mask:
            assert self.args.one_speaker
            y_min, y_max, x_min, x_max = v2['bboxes'][str(active_spk_indices[0])]
            mask = np.zeros(self.image_res)
            mask[y_min: y_max, x_min: x_max] = 1
            sample['tgt_rgb'] = torch.cat([sample['tgt_rgb'], to_tensor(mask).unsqueeze(0)], dim=0)
        if self.args.use_depth:
            sample['src_depth'] = self.process_depth(np.array(Image.open(v1['image'][:-4] + '_depth.png')))
        if self.args.use_depth_mini:
            src_depth = self.process_depth(np.array(Image.open(v1['image'][:-4] + '_depth.png')))
            sample['depth_mini'] = torchvision.transforms.Resize((8, 8))(src_depth).reshape(-1)
        if self.args.use_tgt_depth:
            sample['tgt_depth'] = self.process_depth(np.array(Image.open(v2['image'][:-4] + '_depth.png')))

        sample['tgt_pose'] = to_tensor(tgt_pose)
        sample['tgt_rotation'] = to_tensor(tgt_rotation)

        if self.args.use_src_location or self.args.dereverb_input:
            sample['src_location'] = to_tensor([v1['location'][0], v1['location'][2]])
        if self.args.use_tgt_location:
            sample['tgt_location'] = to_tensor([v2['location'][0], v2['location'][2]])
        if self.args.use_speaker_location:
            speaker = speakers[active_spk_indices[0]]
            sample['speaker_location'] = to_tensor([speaker['location'][0], speaker['location'][2]])
        
        speaker = speakers[active_spk_indices[0]]
        sample['src_tgt_dist'] = np.sqrt((v1['location'][0] - v2['location'][0])** 2 + (v1['location'][2] - v2['location'][2])** 2)
        sample['src_speaker_dist'] = np.sqrt((v1['location'][0] - speaker['location'][0])** 2 + (v1['location'][2] - speaker['location'][2])** 2)
        sample['speaker_tgt_dist'] = np.sqrt((speaker['location'][0] - v2['location'][0])** 2 + (speaker['location'][2] - v2['location'][2])** 2)

        if self.args.use_speaker_bboxes:
            bboxes = np.concatenate([v1['bboxes'][str(active_spk_idx)] for active_spk_idx in active_spk_indices], axis=0)
            bboxes = bboxes / 255.0
            sample['bboxes'] = to_tensor(bboxes)

            if self.args.visualize:
                sample['src_rgb_copy'] = self.draw_bbox_on_image(bboxes, v1['image'])

        if self.args.use_speaker_pose:
            speaker_pose = compute_relative_pose(v1, speakers[active_spk_indices[0]])
            if self.args.encode_sincos:
                sample['speaker_pose'] = to_tensor([speaker_pose[0], np.sin(speaker_pose[1]), np.cos(speaker_pose[1])])
            else:
                sample['speaker_pose'] = to_tensor(speaker_pose)

        if self.args.use_tgt_bboxes:
            tgt_bboxes = np.concatenate([v2['bboxes'][str(active_spk_idx)] for active_spk_idx in active_spk_indices],
                                        axis=0)
            sample['tgt_bboxes'] = to_tensor(tgt_bboxes) / 255.0

        if self.args.use_speaker_pose_wrt_tgt:
            speaker_pose = compute_relative_pose(v2, speakers[active_spk_indices[0]])
            if self.args.encode_sincos:
                sample['speaker_pose_wrt_tgt'] = to_tensor([speaker_pose[0], np.sin(speaker_pose[1]), np.cos(speaker_pose[1])])
            else:
                sample['speaker_pose_wrt_tgt'] = to_tensor(speaker_pose)

        return sample

    def draw_bbox_on_image(self, bboxes, image_file):
        import cv2

        y_min, y_max, x_min, x_max = [int(x) for x in bboxes * 255]
        image = np.array(Image.open(image_file)).astype(np.float32)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return self.process_rgb(image)

    def process_rgb(self, rgb):
        rgb = to_tensor(rgb).permute(2, 0, 1) / 255.0
        if rgb.shape[1:] != self.image_res:
            rgb = torchvision.transforms.Resize(self.image_res)(rgb)

        return rgb

    def process_depth(self, depth):
        depth = to_tensor(depth).unsqueeze(0) / 255.0
        if depth.shape[1:] != self.image_res:
            depth = torchvision.transforms.Resize(self.image_res)(depth)

        return depth

    def process_audio(self, audio):
        if audio.shape[1] < self.audio_len:
            audio = np.pad(audio, ((0, 0), (0, self.audio_len - audio.shape[1])))
        return audio

    def process_rir(self, rir):
        assert len(rir.shape) == 2
        rir_ch0 = np.trim_zeros(rir[0], 'b')
        rir_ch1 = np.trim_zeros(rir[1], 'b')
        print(rir.shape, rir_ch0.shape, rir_ch1.shape)
        max_len = max(rir_ch0.shape[0], rir_ch1.shape[0])
        rir = np.stack([np.pad(rir_ch0, (0, max_len - rir_ch0.shape[0])), np.pad(rir_ch1, (0, max_len - rir_ch1.shape[0]))])
        return rir
