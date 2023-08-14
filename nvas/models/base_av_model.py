# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
import json

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
from torchaudio.transforms import GriffinLim
from torch.optim.lr_scheduler import LambdaLR

from nvas.datasets.replay_nvas_dataset import to_tensor
from nvas.loss import MultiResolutionSTFTLoss, STFTLoss, LogMagSTFTLoss, MultiResolutionNoSCSTFTLoss, \
    MagLogMagSTFTLoss, MagSTFTLoss, WaveL2Loss, RelativeMagSTFTLoss, MagRelativeMagSTFTLoss
from nvas.datasets.replay_nvas_dataset import gcc_phat

RT60_ESTIMATOR = None


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def load_rt60_estimator(device):
    global RT60_ESTIMATOR
    if RT60_ESTIMATOR is None:
        from nvas.models.visual_net import VisualNet
        RT60_ESTIMATOR = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
        pretrained_weights = 'data/models/rt60_estimator.pth'
        RT60_ESTIMATOR.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['predictor'])
        RT60_ESTIMATOR.to(device=device).eval()

    return RT60_ESTIMATOR


def estimate_rt60(estimator, wav):
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, window=torch.hamming_window(400, device=wav.device),
                      pad_mode='constant', return_complex=True)
    spec = torch.log1p(stft.abs()).unsqueeze(1)
    with torch.no_grad():
        estimated_rt60 = estimator(spec)
    return estimated_rt60


class SpecLoss:
    def __init__(self, distance_metric, log1p=False, log10=False):
        self.distance_metric = distance_metric
        self.log1p = log1p
        self.log10 = log10

    def __call__(self, pred, gt):
        if self.log10:
            gt = torch.log10(gt + 1e-5)
            pred = torch.log10(torch.clamp(pred, 1e-5))
        if self.log1p:
            gt = torch.log1p(gt)
            pred = torch.log1p(torch.clamp(pred, 0))

        if self.distance_metric == 'mse':
            loss = F.mse_loss(gt, pred)
        elif self.distance_metric == 'l1':
            loss = F.l1_loss(gt, pred)
        else:
            raise ValueError

        return loss


class BaseAVModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.learnable = True

        self.first_val = True
        self.loss = None
        self.best_val_metric = float('inf')
        self.target_metric = 'mag_distance'
        self.test_stats = defaultdict(list)

    def configure_loss(self):
        if self.args.decode_wav:
            if self.args.multires_stft:
                self.loss = MultiResolutionSTFTLoss(fft_sizes=[int(x) for x in self.args.fft_sizes.split(',')],
                                                    hop_sizes=[int(x) for x in self.args.hop_sizes.split(',')],
                                                    win_lengths=[int(x) for x in self.args.win_lengths.split(',')],
                                                    factor_sc=1, factor_mag=1)
            elif self.args.mag_logmag:
                self.loss = MagLogMagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                              window="hamming_window")
            elif self.args.mag:
                self.loss = MagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                        window="hamming_window")
            elif self.args.rel_mag:
                self.loss = RelativeMagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                                window="hamming_window")
            elif self.args.mag_rel_mag:
                self.loss = MagRelativeMagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                                   window="hamming_window")
            else:
                self.loss = LogMagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                           window="hamming_window")

            if self.args.extra_wave_loss:
                self.extra_wave_loss = WaveL2Loss()
        else:
            # only for magnitude spectrogram based solutions
            self.loss = SpecLoss(self.args.loss, log1p=self.args.log1p, log10=self.args.log10)

    def load_weights(self, ckpt):
        self.load_state_dict(ckpt['state_dict'], strict=True)

    def print(self, *args, **kwargs):
        if self.global_rank == 0:
            logging.info(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)[0]

    def compute_loss(self, outputs):
        stats = {'loss': 0}
        pred, tgt = outputs['pred'], outputs['tgt']

        if self.args.skip_spec_loss:
            return stats

        if self.args.decode_wav and (len(pred.shape) == len(tgt.shape) == 3) and \
                not (self.args.rel_mag or self.args.mag_rel_mag):
            # TODO: rethink the binaural model
            pred = pred.reshape(-1, pred.shape[-1])
            tgt = tgt.reshape(-1, tgt.shape[-1])

        loss = self.loss(pred, tgt)
        if isinstance(loss, tuple):
            stats['mag_loss'] = loss[1]
            stats['sc_loss'] = loss[0]
            stats['loss'] = stats['loss'] + loss[0] + loss[1]
        else:
            stats['mag_loss'] = loss
            stats['loss'] = loss

        if self.args.extra_wave_loss:
            wave_loss = self.extra_wave_loss(pred, tgt)
            stats['wave_loss'] = wave_loss
            stats['loss'] = stats['loss'] + wave_loss * self.args.extra_wave_loss_weight

        return stats

    def run_step(self, batch, batch_idx, phase=None):
        outputs = self.audio_synthesis(batch, batch_idx, phase)
        pred, tgt = outputs['pred'], outputs['tgt']
        stats = self.compute_loss(outputs)

        return stats, pred, tgt

    def training_epoch_end(self, outputs):
        metrics = outputs[0].keys()
        output_str = f'Train epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.Tensor([output[metric] for output in outputs]).mean()
            self.logger.experiment.add_scalar(f'train/{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

        if self.args.scheduler != 'none':
            self.logger.experiment.add_scalar('train/lr', self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0],
                                              self.current_epoch)

        self.print(output_str[:-2])

    def spec2wav(self, spec, input_wav=None):
        if self.args.use_input_phase:
            phase = self.wav2spec(input_wav)[..., -1]
            real_imag = spec.permute(0, 1, 3, 4, 2) * torch.stack([phase.cos(), phase.sin()], dim=-1)
            reshaped_real_imag = real_imag.reshape(-1, *real_imag.shape[2:])  # B, F, T, R
            wav = torch.istft(reshaped_real_imag, n_fft=512, hop_length=self.args.hop_length,
                              win_length=400, window=torch.hamming_window(400, device=spec.device))
            wav = wav.reshape(real_imag.shape[0], real_imag.shape[1], wav.shape[1])
        elif self.args.predict_mask:
            assert self.args.use_real_imag
            reshaped_spec = spec.reshape(-1, *spec.shape[2:])
            wav = torch.istft(reshaped_spec.permute(0, 2, 3, 1), n_fft=512, hop_length=self.args.hop_length,
                              win_length=400, window=torch.hamming_window(400, device=spec.device))
            wav = wav.reshape(spec.shape[0], spec.shape[1], wav.shape[1])
        else:
            wav = []
            for channel in range(spec.shape[1]):
                channel_spec = spec[:, channel]
                channel_wav = GriffinLim(n_fft=512, hop_length=self.args.hop_length, win_length=400,
                                         window_fn=torch.hamming_window, power=1, rand_init=False
                                         )(channel_spec.squeeze(1).cpu()).to(device=self.device)
                wav.append(channel_wav)
            wav = torch.stack(wav, dim=1)

        return wav

    def wav2spec(self, wav):
        if len(wav.shape) == 3:
            b, c = wav.shape[:2]
            wav = wav.reshape((-1, wav.shape[2]))
            reshaped = True

        if self.args.use_real_imag:
            spec = torch.stft(wav, n_fft=512, hop_length=self.args.hop_length, win_length=400,
                              window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                              return_complex=False)
        else:
            stft = torch.stft(wav, n_fft=512, hop_length=self.args.hop_length, win_length=400,
                              window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                              return_complex=True)
            mag = stft.abs()
            phase = stft.angle()
            spec = torch.stack([mag, phase], dim=-1)

        if reshaped:
            # batch x num channel x freq x time x mag/phase (real/imag)
            spec = spec.reshape((b, c) + spec.shape[1:])

        return spec

    def validation_step(self, batch, batch_idx, convolve_rir=False, test=False):
        stats, pred, tgt = self.run_step(batch, batch_idx, phase='val' if not test else 'test')

        pred_wav = pred if self.args.decode_wav else self.spec2wav(pred, batch['src_wav'])
        if self.args.eval_input:
            pred_wav = batch['src_wav']
        tgt_wav = batch['tgt_wav']

        if 'l2_distance' in self.metrics:
            stats['l2_distance'] = (pred_wav - tgt_wav).pow(2).mean([1, 2])

        if 'rt60_error' in self.metrics:
            rt60_estimator = load_rt60_estimator(self.device)
            pred_rt60 = estimate_rt60(rt60_estimator, pred_wav.reshape(-1, pred_wav.shape[-1]))
            tgt_rt60 = estimate_rt60(rt60_estimator, tgt_wav.reshape(-1, tgt_wav.shape[-1]))
            stats['rt60_error'] = (pred_rt60 - tgt_rt60).abs().mean().cpu().numpy()
            # if test:
            #     self.test_stats['pred_rt60'] += pred_rt60.cpu().numpy().tolist()
            #     self.test_stats['tgt_rt60'] += tgt_rt60.cpu().numpy().tolist()

        if 'stft_distance' in self.metrics:
            pred_spec_l, tgt_spec_l = eval_stft(pred_wav[:, 0]), eval_stft(tgt_wav[:, 0])
            pred_spec_r, tgt_spec_r = eval_stft(pred_wav[:, 1]), eval_stft(tgt_wav[:, 1])
            stats['stft_distance'] = (pred_spec_l - tgt_spec_l).pow(2).sum((1, 2, 3)).sqrt() + \
                                     (pred_spec_r - tgt_spec_r).pow(2).sum((1, 2, 3)).sqrt()

        if 'mag_distance' in self.metrics:
            pred_spec_l, tgt_spec_l = eval_mag(pred_wav[:, 0]), eval_mag(tgt_wav[:, 0])
            pred_spec_r, tgt_spec_r = eval_mag(pred_wav[:, 1]), eval_mag(tgt_wav[:, 1])
            # stats['mag_distance'] = (pred_spec_l - tgt_spec_l).pow(2).sum((1, 2)).sqrt() + \
            #                         (pred_spec_r - tgt_spec_r).pow(2).sum((1, 2)).sqrt()
            stats['mag_distance'] = (pred_spec_l - tgt_spec_l).pow(2).sqrt().mean((1, 2)) + \
                                    (pred_spec_r - tgt_spec_r).pow(2).sqrt().mean((1, 2))

        if 'lr_ratio' in self.metrics:
            pred_lr_ratio = 10 * torch.log10(pred_wav[:, 0].pow(2).sum(-1) / (pred_wav[:, 1].pow(2).sum(-1) + 1e-5))
            tgt_lr_ratio = 10 * torch.log10(tgt_wav[:, 0].pow(2).sum(-1) / (tgt_wav[:, 1].pow(2).sum(-1) + 1e-5))
            stats['lr_ratio'] = (pred_lr_ratio - tgt_lr_ratio).abs()

        tgt_wav = tgt_wav.reshape(-1, tgt_wav.shape[-1]).cpu().numpy()
        pred_wav = pred_wav.reshape(-1, pred_wav.shape[-1]).cpu().numpy()

        # calculate speech measurements
        metric_scores = defaultdict(list)
        for speech_metric in self.speech_metric_names:
            metric_scores[speech_metric] = list()
        for i in range(tgt_wav.shape[0]):
            if len(self.speech_metric_names) != 0:
                scores = self.speech_metrics(pred_wav[i], tgt_wav[i], rate=self.args.sample_rate)
                tgt_scores = self.speech_metrics(tgt_wav[i], pred_wav[i], rate=self.args.sample_rate)
                for metric_name in scores:
                    metric_scores[metric_name].append(scores[metric_name])
                    metric_scores[metric_name+'_diff'].append(abs(scores[metric_name] - tgt_scores[metric_name]))
                    metric_scores[metric_name+'_rel'].append(abs(scores[metric_name] - tgt_scores[metric_name]) / tgt_scores[metric_name])
                    # self.test_stats[metric_name+'_diff'].append(abs(scores[metric_name] - tgt_scores[metric_name]))

        # calculate delays between two waveforms
        if 'delay' in self.metrics:
            for pred_channel, tgt_channel in zip(pred_wav, tgt_wav):
                delay = abs(gcc_phat(pred_channel, tgt_channel, self.args.sample_rate, return_second=False)[0])
                delay = np.clip(delay, 0, 50)
                metric_scores['delay'].append(delay)

        for metric_name, metric_value_list in metric_scores.items():
            stats[metric_name] = to_tensor(np.mean(metric_value_list))
        
        if test:
            if 'mag_distance' in self.metrics:
                self.test_stats['mag_distance'] += stats['mag_distance'].cpu().numpy().tolist()
                
            # self.test_stats['src_tgt_dist'] += batch['src_tgt_dist'].cpu().numpy().tolist()
            # self.test_stats['src_speaker_dist'] += batch['src_speaker_dist'].cpu().numpy().tolist()
            # self.test_stats['speaker_tgt_dist'] += batch['speaker_tgt_dist'].cpu().numpy().tolist()

        return stats

    def validation_epoch_end(self, outputs):
        if self.first_val:
            self.first_val = False
            return

        gathered_outputs = self.all_gather(outputs)
        metrics = gathered_outputs[0].keys()
        output_str = f'Val epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.concat([output[metric].unsqueeze(-1) for output in gathered_outputs], dim=1).mean()
            self.logger.experiment.add_scalar(f'val/{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

            # manually save the best model
            if metric == self.target_metric and avg_value < self.best_val_metric:
                self.best_val_metric = avg_value
                if self.global_rank == 0:
                    model_dir = os.path.join(self.args.model_dir, self.args.version)
                    os.makedirs(model_dir, exist_ok=True)
                    ckpt_path = os.path.join(model_dir, f'best_val.ckpt')
                    torch.save({'state_dict': self.state_dict(),
                                'hparams': self.args}, ckpt_path)

        self.print(output_str[:-2])

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, convolve_rir=True, test=True)

    def test_epoch_end(self, outputs):
        gathered_outputs = self.all_gather(outputs)

        metrics = gathered_outputs[0].keys()
        output_str = f'Test epoch {self.current_epoch}, '
        for metric in metrics:
            values = torch.concat([output[metric].unsqueeze(-1) for output in gathered_outputs], dim=1)
            tokens = self.args.from_pretrained.split('=')
            if self.args.test_all and len(tokens) > 1:
                ckpt = int(tokens[-1][:tokens[-1].find('.')])
                self.logger.experiment.add_scalar(f"{self.args.test_split}/{metric}", values.mean(), ckpt)
            output_str += f'{metric}: {values.mean():.4f}, '

            if metric in self.metrics:
                if self.args.print_percent:
                    for percent in [70, 80, 90, 95]:
                        sorted_values = torch.sort(values.squeeze())[0]
                        top_values = sorted_values[: int(sorted_values.shape[0] * percent / 100)]
                        output_str += f'{metric}_top_{percent}%: {top_values.mean():.4f}, '

                if self.args.plot_hist:
                    import matplotlib.pyplot as plt
                    plt.hist(values.cpu().squeeze().numpy(), bins=50)
                    plt.savefig(os.path.join(self.args.model_dir, self.args.version, f'{metric}_hist.png'))
                    plt.close()
        self.print(output_str[:-2].replace('_distance', '').replace('rt60_error', 'rte').replace('lr_ratio_peak', 'lr_peak'))

    def save_test_stats(self):
        postfix = f'_{self.args.test_split}'
        file_path = os.path.join(self.args.model_dir, self.args.version, 'test_stats' + postfix + '.json')
        with open(file_path, 'w') as fo:
            json.dump(self.test_stats, fo)

    def configure_optimizers(self):
        opt_dict = {'optimizer': torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)}

        return opt_dict


def eval_stft(wav):
    # return real and imaginary components as two channels
    assert len(wav.shape) == 2
    spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=False)

    return spec


def eval_mag(wav, log=False):
    assert len(wav.shape) == 2
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=True)
    if log:
        mag = torch.log(stft.abs() + 1e-5)
    else:
        mag = stft.abs()

    return mag
