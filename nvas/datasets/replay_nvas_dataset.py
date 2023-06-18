# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json

import numpy as np
import torch
import torchvision
from decord import VideoReader, AudioReader
from decord import cpu
from PIL import Image
from torch.utils.data import Dataset
import julius
from scipy.io import wavfile

from nvas.datasets.soundspaces_nvas_dataset import butter_bandpass_filter, to_tensor

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)


SCENE_SPLITS = {
    'v3': {
        'train': ['SC-1022', 'SC-1023', 'SC-1024', 'SC-1025', 'SC-1032', 'SC-1033', 'SC-1035', 'SC-1036',
                  'SC-1040', 'SC-1042', 'SC-1045', 'SC-1047', 'SC-1048', 'SC-1049', 'SC-1050', 'SC-1059',
                  'SC-1066', 'SC-1069', 'SC-1070', 'SC-1071', 'SC-1075', 'SC-1076', 'SC-1077', 'SC-1078',
                  'SC-1079', 'SC-1081', 'SC-1085', 'SC-1087', 'SC-1090', 'SC-1091', 'SC-1101', 'SC-1102',
                  'SC-1104', 'SC-1105', 'SC-1106', 'SC-1108'],
        'val': ['SC-1026', 'SC-1054', 'SC-1073', 'SC-1082', 'SC-1092', 'SC-1103'],  # 6
        'test': ['SC-1027', 'SC-1044', 'SC-1052', 'SC-1074', 'SC-1084', 'SC-1093', 'SC-1107']  # 7
    },
}

MEAN_LOCATION = [[188.02005, 161.4408, 154.40367],
                 [465.3129, 271.95114, 154.1191],
                 [410.8833, 693.10876, 154.20988],
                 [48.7043, 546.82465, 154.57697],
                 [78.69773, 281.85303, 184.40979],
                 [340.33405, 167.44298, 131.02776],
                 [491.1604, 521.15546, 190.65074],
                 [208.43025, 712.3111, 146.60208]]
MEAN_ROTATION = [[0.16835392, 0.9591619, -0.27663985],
                 [-0.83045524, 0.43742082, -0.32443687],
                 [-0.25761527, -0.9389267, -0.27287343],
                 [0.9269612, -0.2707318, -0.26748332],
                 [0.6317217, 0.67077196, -0.39229873],
                 [-0.41840369, 0.8836631, -0.19575387],
                 [-0.7555723, -0.48939732, -0.4576416],
                 [0.43345883, -0.8781099, -0.2396668]]

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16, return_second=True):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    if not return_second:
        tau *= fs

    return tau, cc


def read_clip(img_file, split, target_sr):
    try:
        video_file = img_file.replace('.png', '.mp4')
        binaural_file = img_file.replace('.png', '.wav')
        if os.path.exists(img_file):
            rgb = np.array(Image.open(img_file))
        else:
            print('Imag file does not exist: ', img_file)
            vr = VideoReader(video_file, ctx=cpu(0))
            rgb = vr[np.random.randint(0, len(vr)) if split == 'train' else len(vr) // 2].asnumpy()

        input_sr, audio = wavfile.read(binaural_file)
        audio = audio.T
        if input_sr != target_sr:
            audio = julius.resample_frac(torch.tensor(audio), input_sr, target_sr).numpy()
    except Exception as e:
        print(f'Exception {e} happened for {img_file}')
        rgb = np.zeros((720, 1280, 3))
        audio = np.zeros((2, 48000 * 2))

    return rgb, audio


def read_near_audio(clip_dir, target_sr):
    try:
        sr, audio = wavfile.read(os.path.join(clip_dir, 'near.wav'))
        if sr != target_sr:
            audio = julius.resample_frac(torch.tensor(audio), sr, target_sr).numpy()
        audio = np.stack([audio, audio], axis=0)
    except Exception as e:
        print(f"Exception {e} happened for {os.path.join(clip_dir, 'near.wav')}")
        audio = np.zeros((2, 48000 * 2))
    return audio


def xyz_to_angle(xyz):
    return np.array([np.arctan2(xyz[1], xyz[0]), np.arctan2(xyz[2], xyz[1]), np.arctan2(xyz[2], xyz[0])])


def to_array(li):
    return np.array([float(x) for x in li])


def compute_dsp_data(loc, rot, loc2):
    distance = np.sqrt(np.sum(np.array(loc) - np.array(loc2)) ** 2)
    azimuth = np.arctan2(loc2[1] - loc[1], loc2[0] - loc[0]) - rot[0]
    elevation = np.arctan2(loc2[2] - loc[1], loc2[2] - loc[1]) - rot[1]

    return distance, azimuth, elevation


class ReplayNVASDataset(Dataset):
    def __init__(self, split, args):
        super().__init__()
        self.split = split
        self.args = args
        self.rgb_res = (216, 384)
        self.audio_len = args.audio_len
        self.estimated_heads = None

        with open(os.path.join(args.dataset_dir, args.metadata_file), 'r') as fo:
            self.metadata = json.load(fo)
        scene_splits = SCENE_SPLITS[args.dataset_dir.split('/')[-1]]
        self.clip_dirs = [x for x in list(self.metadata.keys()) if x.split('/')[-2] in scene_splits[split]]

        if not args.use_cam_index:
            with open(f'data/appen_dataset/{args.cam_position_file}', 'r') as fo:
                self.camera_positions = json.load(fo)

        print(f'Number of clip is {len(self.clip_dirs)} for {self.split.upper()}')

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, item):
        clip_dir = self.clip_dirs[item]

        files = self.metadata[clip_dir]
        file_1, file_2 = np.random.permutation(files)[:2]
        file_1, file_2 = os.path.join(clip_dir, file_1), os.path.join(clip_dir, file_2)
        v1, a1 = read_clip(file_1, self.split, self.args.sample_rate)
        v2, a2 = read_clip(file_2, self.split, self.args.sample_rate)

        if self.args.input_mono:
            mean = np.mean(a1, axis=0)
            a1 = np.array([mean, mean])

        if self.args.output_mono:
            mean = np.mean(a2, axis=0)
            a2 = np.array([mean, mean])

        if self.args.use_clean_speech:
            a1 = read_near_audio(clip_dir, self.args.sample_rate)

        if self.args.use_clean_as_tgt:
            a2 = read_near_audio(clip_dir, self.args.sample_rate)

        if self.args.highpass_filter:
            a1 = butter_bandpass_filter(a1, 150, self.args.sample_rate // 2 - 1, self.args.sample_rate, order=5).copy()
            a2 = butter_bandpass_filter(a2, 150, self.args.sample_rate // 2 - 1, self.args.sample_rate, order=5).copy()

        src_max = np.max(np.abs(a1))
        if self.args.normalize_src and src_max != 0:
            if self.args.normalize_tgt:
                tgt_max = np.max(np.abs(a2))
                if tgt_max != 0:
                    a1 /= src_max
                    a2 /= tgt_max
            else:
                a1 /= src_max
                a2 /= src_max

        if self.args.remove_delay:
            if self.args.gcc_max_pool:
                delay = int(gcc_phat(np.max(a2, axis=0), np.max(a1, axis=0), self.args.sample_rate,
                                     return_second=False)[0])  # positive value means a2 is ahead of a1 and a2 needs to padded
            else:
                delay = int(gcc_phat(np.mean(a2, axis=0), np.mean(a1, axis=0), self.args.sample_rate,
                                     return_second=False)[0])
            delay = np.clip(delay, a_min=-200, a_max=200)
            if delay > 0:
                a1 = np.pad(a1, ((0, 0), (delay, 0)))[:, :-delay]
            else:
                a1 = np.pad(a1, ((0, 0), (0, -delay)))[:, -delay:]

        sample = dict()
        v1, v2 = self.process_rgb(v1.astype(np.float32)), self.process_rgb(v2.astype(np.float32))
        a1, a2 = self.process_audio(a1), self.process_audio(a2)
        if self.split == 'train' and a1.shape[0] > self.audio_len and a2.shape[0] > self.audio_len:
            random_start = np.random.randint(a1.shape[0] - self.audio_len)
            sample['src_wav'] = to_tensor(a1[:, random_start: random_start + self.audio_len])
            sample['tgt_wav'] = to_tensor(a2[:, random_start: random_start + self.audio_len])
        else:
            sample['src_wav'] = to_tensor(a1[:, :self.audio_len])
            sample['tgt_wav'] = to_tensor(a2[:, :self.audio_len])
        sample['src_rgb'] = to_tensor(v1).permute(2, 0, 1) / 255.0
        sample['tgt_rgb'] = to_tensor(v2).permute(2, 0, 1) / 255.0

        src_index = int(file_1.split('/')[-1][:-4])
        tgt_index = int(file_2.split('/')[-1][:-4])
        scene_index = file_1.split('/')[-3]
        if self.args.use_cam_index:
            src_label, tgt_label = np.zeros(8), np.zeros(8)
            src_label[src_index - 1] = 1
            if not self.args.remove_tgt_index:
                tgt_label[tgt_index - 1] = 1
            sample['cam_index'] = np.concatenate([src_label, tgt_label], axis=0)
        else:
            if self.args.use_avg_location:
                src_pos = np.array(MEAN_LOCATION[src_index - 1])
                src_rot = xyz_to_angle(MEAN_ROTATION[src_index - 1])
                tgt_pos = np.array(MEAN_LOCATION[tgt_index - 1])
                tgt_rot = xyz_to_angle(MEAN_ROTATION[tgt_index - 1])
            else:
                src_pos = to_array(self.camera_positions['centers'][scene_index][f'DSLR-{src_index}'])
                src_rot = to_array(self.camera_positions['rotations'][scene_index][f'DSLR-{src_index}'])
                tgt_pos = to_array(self.camera_positions['centers'][scene_index][f'DSLR-{tgt_index}'])
                tgt_rot = to_array(self.camera_positions['rotations'][scene_index][f'DSLR-{tgt_index}'])
                if 'fixed_rotation' in self.args.cam_position_file:
                    src_rot = xyz_to_angle(src_rot)
                    tgt_rot = xyz_to_angle(tgt_rot)
            if self.args.rotate_rel_pose:
                d = np.sqrt((src_pos - tgt_pos)[0] ** 2 + (src_pos - tgt_pos)[1] ** 2)
                angle = np.arctan2(tgt_pos[1] - src_pos[1], tgt_pos[0] - src_pos[0])
                rel_rot = 90 - (src_rot[0] - angle)
                rel_tgt_pos = np.array([d * np.cos(rel_rot), d * np.sin(rel_rot), tgt_pos[2] - src_pos[2]])
            else:
                rel_tgt_pos = tgt_pos - src_pos
            rel_tgt_rot = tgt_rot - src_rot
            sample['tgt_rotation_degree'] = to_tensor(np.rad2deg(rel_tgt_rot))

            if self.args.encode_sincos:
                rel_tgt_rot = [np.sin(rel_tgt_rot[0]), np.cos(rel_tgt_rot[0]),
                               np.sin(rel_tgt_rot[1]), np.cos(rel_tgt_rot[1]),
                               np.sin(rel_tgt_rot[2]), np.cos(rel_tgt_rot[2])]
            sample['tgt_pose'] = to_tensor(rel_tgt_pos)
            sample['tgt_rotation'] = to_tensor(rel_tgt_rot)

        if self.args.use_speaker_bboxes:
            bboxes = np.load(file_1.replace('.png', '.npy'))
            audio_len = self.audio_len / self.args.sample_rate
            if self.args.use_all_bboxes:
                if self.args.use_local_conditioning:
                    sample['bboxes'] = to_tensor(bboxes[: int(audio_len * 5), :4])
                else:
                    sample['bboxes'] = to_tensor(bboxes[: int(audio_len * 5), :4]).reshape(-1)
            else:
                spk_ids, counts = np.unique(bboxes[:, 4], return_counts=True)
                spk_idx = np.nonzero(bboxes[:, 4] == spk_ids[np.argmax(counts)])[0][0]
                sample['bboxes'] = to_tensor(bboxes[spk_idx][:4])

                if self.args.model == 'dsp':
                    if self.estimated_heads is None:
                        import pickle
                        with open('/checkpoint/romansh/tmp/changan-speaker-locations.pkl', 'rb') as fo:
                            self.estimated_heads = pickle.load(fo)
                    scene_name, clip_idx, file_name = file_1.split('/')[-3:]
                    clip_idx = int(clip_idx)
                    cam_idx = int(file_name[0])
                    assert len(self.estimated_heads[scene_name]) > clip_idx, len(self.estimated_heads[scene_name])
                    # assert cam_idx in self.estimated_heads[scene_name][clip_idx], (scene_name, clip_idx, self.estimated_heads[scene_name][clip_idx])
                    if cam_idx not in self.estimated_heads[scene_name][clip_idx]:
                        assert len(self.estimated_heads[scene_name][clip_idx]) == 3
                        estimated_head = self.estimated_heads[scene_name][clip_idx]
                    else:
                        estimated_head = self.estimated_heads[scene_name][clip_idx][cam_idx]
                    distance, azimuth, elevation = compute_dsp_data(src_pos, src_rot, estimated_head)
                    sample['src_distance'] = to_tensor([distance])
                    sample['src_azimuth'] = to_tensor([distance])
                    sample['src_elevation'] = to_tensor([distance])

                    distance, azimuth, elevation = compute_dsp_data(tgt_pos, tgt_rot, estimated_head)
                    sample['tgt_distance'] = to_tensor([distance])
                    sample['tgt_azimuth'] = to_tensor([distance])
                    sample['tgt_elevation'] = to_tensor([distance])

            if self.args.visualize and not self.args.use_all_bboxes:
                sample['src_rgb_copy'] = self.draw_bbox_on_image(bboxes[spk_idx], file_1)

        if self.args.use_tgt_bboxes:
            bboxes = np.load(file_2.replace('.png', '.npy'))
            spk_ids, counts = np.unique(bboxes[:, 4], return_counts=True)
            spk_idx = np.nonzero(bboxes[:, 4] == spk_ids[np.argmax(counts)])[0][0]
            sample['tgt_bboxes'] = to_tensor(bboxes[spk_idx][:4])
            if self.args.visualize:
                sample['tgt_rgb_copy'] = self.draw_bbox_on_image(bboxes[spk_idx], file_2)

        if self.args.use_src_location:
            sample['src_location'] = to_tensor(self.camera_positions['centers'][scene_index][f'DSLR-{src_index}'])

        if self.args.use_tgt_location:
            sample['tgt_location'] = to_tensor(self.camera_positions['centers'][scene_index][f'DSLR-{tgt_index}'])

        return sample

    def draw_bbox_on_image(self, bboxes, image_file):
        import cv2
        # import matplotlib.pyplot as plt

        x_min = int(bboxes[0] * self.rgb_res[1])
        y_min = int(bboxes[1] * self.rgb_res[0])
        x_max = int(bboxes[2] * self.rgb_res[1])
        y_max = int(bboxes[3] * self.rgb_res[0])
        image = np.array(Image.open(image_file)).astype(np.float32)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # plt.imsave('test.png', image / 255.0)
        image = to_tensor(self.process_rgb(image)).permute(2, 0, 1) / 255.0

        return image

    def process_rgb(self, rgb):
        if rgb.shape[:2] != self.rgb_res:
            rgb = torchvision.transforms.Resize(self.rgb_res)(to_tensor(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()

        return rgb

    def process_audio(self, audio):
        if audio.shape[1] < self.audio_len:
            audio = np.pad(audio, ((0, 0), (0, self.audio_len - audio.shape[1])))
        return audio
