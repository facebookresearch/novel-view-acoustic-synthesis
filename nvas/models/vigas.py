# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import speechmetrics

from nvas.models.base_av_model import BaseAVModel
from nvas.models.binaural_nets import HyperConvWavenet, WaveoutBlock, Warpnet
from nvas.models.vida import create_conv

DEREVERBERATOR = None
DENOISER = None


def update_args(args, hparams):
    for key, value in vars(hparams).items():
        setattr(args, key, value)
    return args


def load_dereverberator(dereverb_model, device):
    global DEREVERBERATOR
    from trainer import parser
    if DEREVERBERATOR is None:
        state_dict = torch.load(dereverb_model, map_location='cpu')
        args = parser.parse_args("")
        update_args(args, state_dict['hparams'])
        DEREVERBERATOR = WaveNet(args).to(device=device)
        DEREVERBERATOR.load_state_dict(state_dict['state_dict'])
        DEREVERBERATOR.eval()
        print('Loaded dereverberator')

    return DEREVERBERATOR


def load_denoiser(device):
    global DENOISER
    if DENOISER is None:
        from denoiser import pretrained
        DENOISER = pretrained.dns64().to(device)
        DENOISER.eval()
        print('Loaded denoiser')

    return DENOISER


def separate_binaural(batch):
    binaural_audio = batch['tgt_wav']
    for key, value in batch.items():
        if key != 'tgt_wav':
            batch[key] = torch.concat([value, value], dim=0)

    batch['tgt_wav'] = torch.concat([binaural_audio[:, 0:1].expand_as(binaural_audio),
                                     binaural_audio[:, 1:2].expand_as(binaural_audio)], dim=0)
    batch['ear_index'] = torch.concat([torch.zeros(binaural_audio.shape[0], 1),
                                      torch.ones(binaural_audio.shape[0], 1)], dim=0).to(binaural_audio.device)
    return batch


def merge_binaural(batch, pred_wav):
    binaural_audio = batch['tgt_wav']
    batch_size = binaural_audio.shape[0] // 2
    for key, value in batch.items():
        if key != 'tgt_wav':
            batch[key] = batch[key][:batch_size]

    batch['tgt_wav'] = torch.concat([binaural_audio[:batch_size, 0:1],
                                     binaural_audio[batch_size:, 1:2]], dim=1)

    pred_wav = torch.concat([pred_wav[:batch_size, 0:1], pred_wav[batch_size:, 1:2]], dim=1)

    return batch, pred_wav


class ViGAS(BaseAVModel):
    def __init__(self, args):
        super(ViGAS, self).__init__(args)
        self.args = args

        non_learnable_feat_dim = sum([args.use_tgt_pose, args.use_speaker_pose, args.use_speaker_pose_wrt_tgt]) * 2 + \
                                 (20 if args.use_all_bboxes and not args.use_local_conditioning else 4) * args.use_speaker_bboxes + \
                                 args.use_tgt_rotation + 64 * args.use_depth_mini + \
                                 4 * args.use_tgt_bboxes + 2 * args.use_src_location + 2 * args.use_tgt_location + \
                                 args.index_binaural + 16 * args.use_cam_index

        if args.encode_sincos:
            non_learnable_feat_dim += sum([args.use_tgt_pose, args.use_speaker_pose, args.use_tgt_rotation, args.use_speaker_pose_wrt_tgt])
        if args.dataset == 'appen' and not args.use_cam_index and not (non_learnable_feat_dim == 0):
            non_learnable_feat_dim += 4
        if non_learnable_feat_dim == 0:
            non_learnable_feat_dim = 1
        view_dim = 256 if args.use_rgb or args.mlp_fusion else non_learnable_feat_dim
        wavenet_blocks = args.wavenet_blocks
        layers_per_block = args.layers_per_block
        wavenet_channels = 64

        feat = 512 if self.args.dataset == 'synthetic' else 672
        if args.use_rgb or args.use_tgt_rgb or args.use_tgt_rgb_pano:
            layers = list(torchvision.models.resnet18(pretrained=True).children())[0:-2]
            if args.use_tgt_bboxes_as_mask:
                conv2d = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.visual_encoder = nn.Sequential(conv2d, *layers[1:])
            else:
                self.visual_encoder = nn.Sequential(*layers)
            self.conv1x1 = create_conv(512, 8, 1, 0)

            input_dim = feat * (args.use_rgb + args.use_tgt_rgb) + 864 * args.use_tgt_rgb_pano + non_learnable_feat_dim
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, view_dim)
            )
        elif args.use_depth:
            assert not (self.args.use_rgb or self.args.use_tgt_rgb or self.args.use_tgt_depth)
            conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers = list(torchvision.models.resnet18(pretrained=True).children())[0:-2]
            self.visual_encoder = nn.Sequential(conv1, *layers[1:])
            self.conv1x1 = create_conv(512, 8, 1, 0)

            input_dim = feat + non_learnable_feat_dim
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, view_dim)
            )
        elif args.mlp_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(non_learnable_feat_dim, 512),
                nn.ReLU(),
                nn.Linear(512, view_dim)
            )
        else:
            self.fusion = torch.nn.Identity()

        if args.use_warper:
            self.warper = Warpnet(view_dim=view_dim)

        if args.scale_mag:
            self.mag_scaler = nn.Sequential(
                nn.Linear(view_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )

        if args.wavenet_encoder == 'hyperconv':
            self.input = nn.Conv1d(2, wavenet_channels, kernel_size=1)
            self.input.weight.data.uniform_(-np.sqrt(6.0 / 2), np.sqrt(6.0 / 2))
            self.hyperconv_wavenet = HyperConvWavenet(view_dim, wavenet_channels, wavenet_blocks, layers_per_block,
                                                      remove_hyperconv=args.remove_hyperconv,
                                                      conv_len=3 if args.acausal else 2,
                                                      acausal=args.acausal)
            output_channel = 1 if args.output_mono and args.single_channel else 2
            self.output_net = nn.ModuleList([WaveoutBlock(wavenet_channels, output_channel=output_channel)
                                            for _ in range(wavenet_blocks*layers_per_block)])
        else:
            raise NotImplementedError

        if args.use_spatial_classification_loss:
            self.audio_feat_net = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
            )
            self.audio_visual_fusion = nn.Linear(1024, 512)
            self.spatial_classifier = nn.Linear(512, 1)

        if args.use_viewpoint_contrastive_loss:
            self.audio_feat_net = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
            )
            self.triplet_loss = nn.TripletMarginLoss(args.viewpoint_triplet_margin)

        self.first_val = True
        self.configure_loss()
        self.metrics = ['stft_distance', 'mag_distance', 'delay', 'lr_ratio', 'l2_distance', 'rt60_error']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def compute_visual_feat(self, rgb):
        feat = self.visual_encoder(rgb)
        feat = self.conv1x1(feat).reshape(feat.shape[0], -1)
        return feat

    def forward(self, batch):
        features = []
        if self.args.use_rgb:
            features.append(self.compute_visual_feat(batch['src_rgb']))
        if self.args.use_depth:
            features.append(self.compute_visual_feat(batch['src_depth']))
        if self.args.use_depth_mini:
            features.append(batch['depth_mini'])
        if self.args.use_tgt_rgb:
            features.append(self.compute_visual_feat(batch['tgt_rgb']))
        if self.args.use_tgt_rgb_pano:
            features.append(self.compute_visual_feat(batch['tgt_rgb_pano']))
        if self.args.use_tgt_pose:
            features.append(batch['tgt_pose'])
        if self.args.use_tgt_rotation:
            features.append(batch['tgt_rotation'])
        if self.args.use_speaker_pose:
            features.append(batch['speaker_pose'])
        if self.args.use_speaker_pose_wrt_tgt:
            features.append(batch['speaker_pose_wrt_tgt'])
        if self.args.use_speaker_bboxes and not self.args.use_local_conditioning:
            features.append(batch['bboxes'])
        if self.args.use_tgt_bboxes:
            features.append(batch['tgt_bboxes'])
        if self.args.use_src_location:
            features.append(batch['src_location'])
        if self.args.use_tgt_location:
            features.append(batch['tgt_location'])
        if self.args.index_binaural:
            features.append(batch['ear_index'])
        if self.args.use_cam_index:
            features.append(batch['cam_index'])

        if len(features) == 0:
            features = [torch.zeros((1, 1)).to(self.device)]
        if self.args.use_speaker_bboxes and self.args.use_local_conditioning:
            B, NUM_BBOX, _ = batch['bboxes'].shape
            features = torch.concat(features, dim=1)
            features = features.unsqueeze(1).expand((B, NUM_BBOX, features.shape[-1]))
            features = torch.concat([features, batch['bboxes']], dim=-1)
            features = self.fusion(features.reshape(-1, features.shape[-1])).reshape((B, NUM_BBOX, -1)).permute(0, 2, 1)
        else:
            features = self.fusion(torch.concat(features, dim=1))

        if self.args.dereverb_input:
            dereverberator = load_dereverberator(self.args.dereverb_model, self.device)

            with torch.no_grad():
                batch['src_wav_copy'] = batch['src_wav']
                batch['src_wav'] = dereverberator.audio_synthesis(batch, 0)['pred']

        if self.args.denoise:
            denoiser = load_denoiser(self.device)

            with torch.no_grad():
                batch['src_wav_copy'] = batch['src_wav']
                src_shape = batch['src_wav'].shape
                tgt_shape = (-1, batch['src_wav'].shape[-1])
                batch['src_wav'] = denoiser(batch['src_wav'].reshape(tgt_shape)).reshape(src_shape)
                batch['tgt_wav'] = denoiser(batch['tgt_wav'].reshape(tgt_shape)).reshape(src_shape)

        x = batch['src_wav']
        if self.args.use_warper:
            x = self.warper(x, features.unsqueeze(-1))

        # B x 2 X T
        scale = self.mag_scaler(features) if self.args.scale_mag else torch.tensor([1.0]).to(self.device)
        x = x * scale.unsqueeze(-1)

        if self.args.wavenet_encoder == 'identity':
            x = [x, x]
        elif self.args.wavenet_encoder == 'hyperconv':
            x = self.input(x)
            output, skips = self.hyperconv_wavenet(x, features.unsqueeze(-1) if len(features.shape) == 2 else features)
            # collect output and skips after each layer, "output": x[0], "intermediate": x[1:]
            x = []
            if self.args.use_last_output:
                x = [self.output_net[-1](output), self.output_net[-1](output)]
            else:
                for k in range(len(skips), 0, -1):
                    y = torch.mean(torch.stack(skips[:k], dim=0), dim=0) if not self.args.no_stacking else skips[k-1]
                    y = self.output_net[k-1](y)
                    x += [y]

        return x[0], x[1:], scale

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        if self.args.index_binaural:
            batch = separate_binaural(batch)

        pred_wav, intermediate_wav, scale = self.forward(batch)

        if self.args.subtract_mean:
            pred_wav -= torch.mean(pred_wav, dim=-1, keepdim=True)

        if self.args.output_mono and self.args.single_channel:
            pred_wav = torch.concat([pred_wav, pred_wav], dim=1)

        if self.args.index_binaural:
            batch, pred_wav = merge_binaural(batch, pred_wav)

        if phase != 'train' and not self.args.use_clean_as_tgt:
            tgt_wav = batch['tgt_wav']
            if (pred_wav[:, 0:1] + pred_wav[:, 1:2]).pow(2).sum() < (pred_wav[:, 0:1] - pred_wav[:, 1:2]).pow(2).sum():
                if (pred_wav[:, 0:1] - tgt_wav[:, 0:1]).pow(2).sum() < (pred_wav[:, 0:1] + tgt_wav[:, 0:1]).pow(2).sum():
                    pred_wav = torch.concat([pred_wav[:, 0:1], -pred_wav[:, 1:2]], dim=1)
                else:
                    pred_wav = torch.concat([-pred_wav[:, 0:1], pred_wav[:, 1:2]], dim=1)

        return {'pred': pred_wav, 'tgt': batch['tgt_wav'], 'intermediate': intermediate_wav, 'batch': batch,
                'scale': scale}

    def compute_audio_feat(self, batch):
        window = torch.hamming_window(400, device=self.device)
        input_wav = batch['src_wav'].reshape(-1, *batch['src_wav'].shape[2:])
        audio_spec = torch.stft(input_wav, n_fft=512, hop_length=160, win_length=400, window=window,
                                pad_mode='constant', center=True, return_complex=True).abs()
        audio_spec = audio_spec.reshape(-1, 2, *audio_spec.shape[1:])
        audio_feat = self.audio_feat_net(audio_spec).squeeze(-1).squeeze(-1)

        return audio_feat

    def compute_loss(self, outputs):
        stats = super().compute_loss(outputs)
        batch = outputs['batch']
        pred, tgt = outputs['pred'], outputs['tgt']

        if self.args.use_intermediate_loss:
            tgt = tgt.reshape(-1, tgt.shape[-1])
            intermediate = outputs['intermediate']
            for pred in intermediate:
                pred = pred.reshape(-1, pred.shape[-1])
                loss = self.loss(pred, tgt)
                stats['loss'] = stats['loss'] + loss

        if self.args.use_spatial_classification_loss:
            audio_feat = self.compute_audio_feat(batch)

            rgb = batch['src_rgb']
            new_rgb = []
            flipped = np.zeros([rgb.shape[0]])
            for i in range(rgb.shape[0]):
                if np.random.random() > 0.5:
                    new_rgb.append(torch.flip(rgb[i], dims=[2]))
                    flipped[i] = 1
                else:
                    new_rgb.append(rgb[i])
                    flipped[i] = 0
            new_rgb = torch.stack(new_rgb, dim=0)
            batch['src_rgb'] = new_rgb

            # if flipped:
            #     import matplotlib.pyplot as plt
            #     plt.imsave('flipped.png', rgb[0].permute(1, 2, 0).cpu().numpy())
            #     exit()

            visual_feat = self.compute_visual_feat(new_rgb)
            fused_feature = self.audio_visual_fusion(torch.concat([visual_feat, audio_feat], dim=1))
            binary_pred = self.spatial_classifier(F.relu(fused_feature))

            tgt_label = torch.tensor(flipped).unsqueeze(-1).float().to(self.device)
            binary_loss = F.binary_cross_entropy_with_logits(binary_pred, tgt_label)
            stats['binary_loss'] = binary_loss
            stats['binary_accuracy'] = ((binary_pred > 0.5) == tgt_label).float()
            stats['loss'] = stats['loss'] + binary_loss * self.args.spatial_classification_loss_weight

        if self.args.use_viewpoint_contrastive_loss:
            audio_feat = self.compute_audio_feat(batch)
            pos_visual_feat = self.compute_visual_feat(batch['src_rgb'])
            neg_visual_feat = self.compute_visual_feat(batch['tgt_rgb'])

            triplet_loss = self.triplet_loss(audio_feat, pos_visual_feat, neg_visual_feat)
            stats['triplet_loss'] = triplet_loss
            stats['loss'] = stats['loss'] + triplet_loss * self.args.viewpoint_contrastive_loss_weight

        if self.args.use_scale_loss:
            if self.args.use_peak_scale:
                src_max = batch['src_wav'].abs().max(dim=-1)[0]
                tgt_max = batch['tgt_wav'].abs().max(dim=-1)[0]
            elif self.args.use_rms_scale:
                src_max = batch['src_wav'].pow(2).mean(dim=-1).sqrt()
                tgt_max = batch['tgt_wav'].pow(2).mean(dim=-1).sqrt()
            else:
                raise ValueError

            gt_scale = torch.clip(tgt_max / (src_max + 1E-5), 0, 100)
            scale_loss = F.mse_loss(gt_scale, outputs['scale'])
            stats['scale_loss'] = scale_loss
            stats['loss'] = stats['loss'] + scale_loss * self.args.scale_loss_weight

        return stats
