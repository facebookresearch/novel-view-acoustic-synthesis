# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json

import torch
import numpy as np
import speechmetrics
from scipy.io import wavfile
from scipy.signal import fftconvolve

from nvas.models.base_av_model import BaseAVModel


class Retrieval(BaseAVModel):
    def __init__(self, args):
        super(Retrieval, self).__init__(args)
        self.args = args
        self.learnable = False

        data_dir = f'{args.dataset_dir}/train'
        scenes = os.listdir(data_dir)
        self.rir_list = {}
        for scene in scenes:
            metadata_file = os.path.join(data_dir, scene, args.metadata_file)

            if not os.path.exists(metadata_file):
                continue

            with open(metadata_file, 'r') as fo:
                metadata_list = json.load(fo)
                for metadata in metadata_list:
                    for viewpoint in metadata['viewpoints']:
                        for speaker in metadata['speakers']:
                            self.rir_list[(tuple(viewpoint['location']), tuple(speaker['location']))] = \
                                viewpoint['mono_rir'][str(speaker['id'])]

        self.first_val = True
        self.configure_loss()
        self.metrics = ['stft_distance', 'mag_distance', 'delay', 'lr_ratio', 'lr_ratio_peak', 'l2_distance']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        assert self.args.use_clean_speech
        src_locations = batch['speaker_location'].cpu().numpy()
        tgt_locations = batch['tgt_location'].cpu().numpy()
        src_wavs = batch['src_wav'].cpu().numpy()
        pred_wavs = []
        for i, (src_loc, tgt_loc, src_wav) in enumerate(zip(src_locations, tgt_locations, src_wavs)):
            min_dist = float('inf')
            for (view_loc, speaker_loc), rir_file in self.rir_list.items():
                src_dist = ((src_loc[0] - speaker_loc[0]) ** 2 + (src_loc[1] - speaker_loc[2]) ** 2) ** 0.5
                tgt_dist = ((tgt_loc[0] - view_loc[0]) ** 2 + (tgt_loc[1] - view_loc[2]) ** 2) ** 0.5
                dist = src_dist + tgt_dist
                if dist < min_dist:
                    min_rir_file = rir_file
                    min_dist = dist
                    # min_src_dist = src_dist
                    # min_tgt_dist = tgt_dist
            sr, rir = wavfile.read(min_rir_file)
            # print(min_src_dist, min_tgt_dist)
            pred_wavs.append(np.array([fftconvolve(wav, rir) for wav in src_wav])[:, :src_wav.shape[1]])

        pred_wav = torch.tensor(np.stack(pred_wavs)).to(device=self.device, dtype=torch.float)

        return {'pred': pred_wav, 'tgt': batch['tgt_wav'], 'batch': batch}
