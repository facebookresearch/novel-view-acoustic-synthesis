# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sofa
import numpy as np
import torch as th
import torchaudio as ta
import torch
from torch import nn
import speechmetrics

from novas.models.base_av_model import BaseAVModel


class DSP(BaseAVModel):
    def __init__(self, args):
        super(DSP, self).__init__(args)
        self.args = args
        self.learnable = False

        self.hrtf = Hrtf()

        self.first_val = True
        self.configure_loss()
        self.metrics = ['stft_distance', 'mag_distance', 'delay', 'lr_ratio', 'lr_ratio_peak', 'l2_distance',
                        'rt60_error']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        pred_wav = []
        for i in range(batch['src_wav'].shape[0]):
            if self.args.dataset == 'synthetic':
                speaker_pose = batch['speaker_pose'][i].cpu().numpy()
                speaker_pose_wrt_tgt = batch['speaker_pose_wrt_tgt'][i].cpu().numpy()
                mono_wav = self.hrtf.apply_inv_hrtf(batch['src_wav'][i].cpu() * self.args.base_gain,
                                                    azimuth=np.rad2deg(speaker_pose[1]), elevation=0, distance=speaker_pose[0])
                pred_wav.append(self.hrtf.apply_hrtf(mono_wav, azimuth=np.rad2deg(speaker_pose_wrt_tgt[1]),
                                elevation=0, distance=speaker_pose_wrt_tgt[0]))
            else:
                mono_wav = self.hrtf.apply_inv_hrtf(batch['src_wav'][i].cpu() * self.args.base_gain,
                                                    azimuth=np.rad2deg(batch['src_azimuth'][i].cpu().numpy()),
                                                    elevation=np.rad2deg(batch['src_elevation'][i].cpu().numpy()),
                                                    distance=batch['src_distance'][i].cpu().numpy())
                pred_wav.append(self.hrtf.apply_hrtf(mono_wav,
                                                     azimuth=np.rad2deg(batch['tgt_azimuth'][i].cpu().numpy()),
                                                     elevation=np.rad2deg(batch['tgt_elevation'][i].cpu().numpy()),
                                                     distance=batch['tgt_distance'][i].cpu().numpy()))
        pred_wav = torch.stack(pred_wav, dim=0).to(self.device)[:, :, :self.args.audio_len]

        return {'pred': pred_wav, 'tgt': batch['tgt_wav'], 'batch': batch}


class Hrtf:
    def __init__(
        self,
        hrtf_file: str = "data/KEMAR_hrtf.sofa"
    ):
        hrtf = sofa.Database.open(hrtf_file)
        pos = hrtf.Source.Position.get_values().astype(np.int64)
        self.az, self.el = pos[:, 0], pos[:, 1]
        # filters
        fltrs = hrtf.Data.IR.get_values().astype(np.float32)
        self.filters = {(int(self.az[i]), int(self.el[i])): th.from_numpy(fltrs[i]) for i in range(fltrs.shape[0])}
        # inverse filters
        self.inv_filters = {k: self._invert_fltr(v) for k, v in self.filters.items()}

    def _invert_fltr(self, h):
        H = th.fft.fft(h)
        H_inv = th.conj(H) / (th.abs(H) ** 2 + 1e-4)  # Wiener filter, 1e-4 is a noise estimate to compensate for low energy elements
        h_inv = th.real(th.fft.ifft(H_inv))
        return h_inv

    def _get_fltr_idx(self, azimuth, elevation):
        azimuth = self.az[np.abs(self.az - azimuth).argmin()]
        elevation = self.el[np.abs(self.el - elevation).argmin()]
        return (azimuth, elevation)

    def apply_hrtf(self, mono_signal: th.Tensor, azimuth: float, elevation: float, distance: float, base_gain: float = 1.0):
        """
        mono_signal: 1 x T tensor
        azimuth: the azimuth in degrees between 0 and 360
        elevation: the elevation in degrees between -90 and 90
        distance: the distance in meters
        returns a 2 x T tensor containing the binauralized signal
        """
        # binauralize
        h = self.filters[self._get_fltr_idx(azimuth, elevation)]
        h = th.flip(h, dims=(-1,))
        mono_signal = th.cat([mono_signal, mono_signal], dim=0).unsqueeze(0)  # duplicate signal and add batch dimension
        mono_signal = th.nn.functional.pad(mono_signal, pad=(h.shape[-1], 0))
        binaural = th.nn.functional.conv1d(mono_signal, h.unsqueeze(1), groups=2).squeeze(0)
        # adjust gain based on distance (HRTF is measured at 2m distance but gain information is lost due to normalization)
        ref_dist = 0.2
        dist = max(ref_dist, distance)
        gain = base_gain * ref_dist / dist
        return binaural * gain

    def apply_inv_hrtf(self, binaural_signal: th.Tensor, azimuth: float, elevation: float, distance: float, base_gain: float = 1.0):
        """
        binaural_signal: 2 x T tensor
        azimuth: the azimuth in degrees between 0 and 360
        elevation: the elevation in degrees between -90 and 90
        distance: the distance in meters
        returns a 1 x T tensor containing the mono signal as the mixture of left ear and right ear inverse transformation
        """
        h = self.inv_filters[self._get_fltr_idx(azimuth, elevation)]
        h = th.flip(h, dims=(-1,))
        binaural_signal = binaural_signal.unsqueeze(0)  # add batch dimension
        binaural_signal = th.nn.functional.pad(binaural_signal, pad=(h.shape[-1], 0))
        mono = th.nn.functional.conv1d(binaural_signal, h.unsqueeze(1), groups=2).squeeze(0)
        mono = th.mean(mono, dim=0, keepdim=True)
        # adjust gain based on distance (HRTF is measured at 2m distance but gain information is lost due to normalization)
        ref_dist = 0.2
        dist = max(ref_dist, distance)
        gain = base_gain * ref_dist / dist
        return mono / gain


if __name__ == '__main__':
    ### EXAMPLE ###
    mono_signal, sr = ta.load("speech.wav")
    hrtf = Hrtf()
    binaural_signal = hrtf.apply_hrtf(mono_signal, azimuth=61, elevation=31, distance=1.0)
    mono_recon = hrtf.apply_inv_hrtf(binaural_signal, azimuth=61, elevation=31, distance=1.0)
    ta.save("binaural.wav", binaural_signal, sr)
    ta.save("mono_reconstruction.wav", mono_recon, sr)

