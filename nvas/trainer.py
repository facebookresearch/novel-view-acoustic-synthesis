# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shutil
import logging
import glob
import os
import random
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import torch
import numpy
from tqdm import tqdm
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
seed_everything(1)


parser = argparse.ArgumentParser()
parser.add_argument("--run-type", choices=["train", "eval"], default='train')
parser.add_argument("--model-dir", default='data/models')
parser.add_argument("--dataset-dir", default='data/synthetic_dataset/v1')
parser.add_argument("--metadata-file", default='metadata.json')
parser.add_argument("--cam-position-file", default='camera_positions.json')
parser.add_argument("--version", default='v1')
parser.add_argument("--eval-best", default=False, action='store_true')
parser.add_argument("--eval-last", default=False, action='store_true')
parser.add_argument("--auto-resume", default=False, action='store_true')
parser.add_argument("--overwrite", default=False, action='store_true')
parser.add_argument("--use-rgb", default=False, action='store_true')
parser.add_argument("--use-depth", default=False, action='store_true')
parser.add_argument("--num-channel", default=1, type=int)
parser.add_argument("--n-gpus", default=1, type=int)
parser.add_argument("--num-node", default=1, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--max-epochs", default=60, type=int)
parser.add_argument("--ckpt-interval", default=10, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--wd", default=0, type=float)
parser.add_argument("--scheduler", default='none', type=str)
parser.add_argument("--dataset", default='replay_nvas', type=str)
parser.add_argument("--log-mag", default=False, action='store_true')
parser.add_argument("--no-mask", default=False, action='store_true')
parser.add_argument("--log1p", default=False, action='store_true')
parser.add_argument("--log10", default=False, action='store_true')
parser.add_argument("--save-ckpt-interval", default=1, type=int)
parser.add_argument("--from-pretrained", default='', type=str)
parser.add_argument("--slurm", default=False, action='store_true')
parser.add_argument("--gpu-mem32", default=False, action='store_true')
parser.add_argument("--part", default='learnlab,learnfair', type=str)
parser.add_argument("--test", default=False, action='store_true')
parser.add_argument("--test-all", default=False, action='store_true')
parser.add_argument("--test-split", default='test', type=str)
parser.add_argument("--visualize", default=False, action='store_true')
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--loss", default='l1', type=str)
parser.add_argument("--fast-dev-run", default=False, action='store_true')
parser.add_argument("--progress-bar", "--pb", default=False, action='store_true')
parser.add_argument("--eval-ckpt", default=-1, type=int)
parser.add_argument("--model", default='vigas', type=str)
parser.add_argument("--decode-wav", default=False, action='store_true')
parser.add_argument("--encode-wav", default=False, action='store_true')
parser.add_argument("--hop-length", default=160, type=int)
parser.add_argument("--sample-rate", default=16000, type=int)
parser.add_argument("--multires-stft", default=False, action='store_true')
parser.add_argument("--multires-stft-no-sc", default=False, action='store_true')
parser.add_argument("--mag-logmag", default=False, action='store_true')
parser.add_argument("--spectral-stft", default=False, action='store_true')
parser.add_argument("--remove-mel-loss", default=False, action='store_true')
parser.add_argument("--fft-sizes", default='256,512,1024', type=str)
parser.add_argument("--hop-sizes", default='64,128,256', type=str)
parser.add_argument("--win-lengths", default='128,400,600', type=str)
parser.add_argument("--stft-loss-weight", default=45, type=int)
parser.add_argument("--encoder-ratios", default='8,4,2,2', type=str)
parser.add_argument("--decoder-ratios", default='8,4,2,2', type=str)
parser.add_argument("--encoder-residual-layers", default=3, type=int)
parser.add_argument("--decoder-residual-layers", default=3, type=int)
parser.add_argument("--ngf", default=32, type=int)
parser.add_argument("--predict-mask", default=False, action='store_true')
parser.add_argument("--use-real-imag", default=False, action='store_true')
parser.add_argument("--num-worker", default=10, type=int)
parser.add_argument("--eval-input", default=False, action='store_true')
parser.add_argument("--comment", default="", type=str)
parser.add_argument("--prev-ckpt", default=-1, type=int)
parser.add_argument("--use-tgt-rgb", default=False, action='store_true')
parser.add_argument("--use-depth-mini", default=False, action='store_true')
parser.add_argument("--use-tgt-depth", default=False, action='store_true')
parser.add_argument("--use-tgt-pose", default=False, action='store_true')
parser.add_argument("--use-input-phase", default=False, action='store_true')
parser.add_argument("--mag", default=False, action='store_true')
parser.add_argument("--rel-mag", default=False, action='store_true')
parser.add_argument("--mag-rel-mag", default=False, action='store_true')
parser.add_argument("--one-speaker", default=False, action='store_true')
parser.add_argument("--use-speaker-pose", default=False, action='store_true')
parser.add_argument("--use-speaker-pose-wrt-tgt", default=False, action='store_true')
parser.add_argument("--use-intermediate-loss", default=False, action='store_true')
parser.add_argument("--remove-delay", default=False, action='store_true')
parser.add_argument("--remove-hyperconv", default=False, action='store_true')
parser.add_argument("--audio-len", default=16000, type=int)
parser.add_argument("--kernel-size", default=400, type=int)
parser.add_argument("--wavenet-blocks", default=3, type=int)
parser.add_argument("--layers-per-block", default=10, type=int)
parser.add_argument("--subtract-mean", default=False, action='store_true')
parser.add_argument("--use-same-speech", default=False, action='store_true')
parser.add_argument("--use-spatial-classification-loss", default=False, action='store_true')
parser.add_argument("--spatial-classification-loss-weight", default=1, type=float)
parser.add_argument("--use-viewpoint-contrastive-loss", default=False, action='store_true')
parser.add_argument("--viewpoint-triplet-margin", default=1, type=float)
parser.add_argument("--viewpoint-contrastive-loss-weight", default=1, type=float)
parser.add_argument("--scale-mag", default=False, action='store_true')
parser.add_argument("--use-scale-loss", default=False, action='store_true')
parser.add_argument("--use-peak-scale", default=False, action='store_true')
parser.add_argument("--use-rms-scale", default=False, action='store_true')
parser.add_argument("--scale-loss-weight", default=1, type=float)
parser.add_argument("--input-mono", default=False, action='store_true')
parser.add_argument("--output-mono", default=False, action='store_true')
parser.add_argument("--randomize-speech-amplitude", default=False, action='store_true')
parser.add_argument("--normalize-input", default=False, action='store_true')
parser.add_argument("--fix-tgt-pose", default=False, action='store_true')
parser.add_argument("--gradient-clip-val", default=0, type=float)
parser.add_argument("--plot-hist", default=False, action='store_true')
parser.add_argument("--print-percent", default=False, action='store_true')
parser.add_argument("--acausal", default=False, action='store_true')
parser.add_argument("--use-speaker-bboxes", default=False, action='store_true')
parser.add_argument("--use-tgt-rotation", default=False, action='store_true')
parser.add_argument("--use-warper", default=False, action='store_true')
parser.add_argument("--encode-sincos", default=False, action='store_true')
parser.add_argument("--wave-l2-loss", default=False, action='store_true')
parser.add_argument("--extra-wave-loss", default=False, action='store_true')
parser.add_argument("--extra-wave-loss-weight", default=1, type=float)
parser.add_argument("--mlp-fusion", default=False, action='store_true')
parser.add_argument("--scale-loss-only", default=False, action='store_true')
parser.add_argument("--remove-wavenet", default=False, action='store_true')
parser.add_argument("--skip-spec-loss", default=False, action='store_true')
parser.add_argument("--use-clean-speech", default=False, action='store_true')
parser.add_argument("--use-clean-as-tgt", default=False, action='store_true')
parser.add_argument("--use-tgt-bboxes", default=False, action='store_true')
parser.add_argument("--use-tgt-bboxes-as-mask", default=False, action='store_true')
parser.add_argument("--use-tgt-rgb-pano", default=False, action='store_true')
parser.add_argument("--no-stacking", default=False, action='store_true')
parser.add_argument("--use-last-output", default=False, action='store_true')
parser.add_argument("--wavenet-encoder", default='hyperconv', type=str,
                    choices=['hyperconv', 'identity'])
parser.add_argument("--use-src-location", default=False, action='store_true')
parser.add_argument("--use-tgt-location", default=False, action='store_true')
parser.add_argument("--use-speaker-location", default=False, action='store_true')
parser.add_argument("--no-prev-reverb", default=False, action='store_true')
parser.add_argument("--highpass-filter", default=False, action='store_true')
parser.add_argument("--single-channel", default=False, action='store_true')
parser.add_argument("--dereverb-input", default=False, action='store_true')
parser.add_argument("--dereverb-model", default='data/models/wavenet_synthetic_v16_mono_rgb_mag_speaker_'
                                                'pose_output_mono_no_delay_highpass_reg_conv_acausal_src_loc_dereverb_'
                                                'single_channel/best_val.ckpt')
parser.add_argument("--index-binaural", default=False, action='store_true')
parser.add_argument("--visualize-bboxes", default=False, action='store_true')
parser.add_argument("--use-cam-index", default=False, action='store_true')
parser.add_argument("--normalize-src", default=False, action='store_true')
parser.add_argument("--normalize-tgt", default=False, action='store_true')
parser.add_argument("--gcc-max-pool", default=False, action='store_true')
parser.add_argument("--remove-tgt-index", default=False, action='store_true')
parser.add_argument("--nn-tgt-weight", default=1, type=float)
parser.add_argument("--base-gain", default=1, type=float)
parser.add_argument("--use-avg-location", default=False, action='store_true')
parser.add_argument("--rotate-rel-pose", default=False, action='store_true')
parser.add_argument("--denoise", default=False, action='store_true')
parser.add_argument("--use-all-bboxes", default=False, action='store_true')
parser.add_argument("--use-local-conditioning", default=False, action='store_true')
parser.add_argument("--eval-difficult", default=False, action='store_true')


def seed_worker(worker_id):
    numpy.random.seed(worker_id)
    random.seed(worker_id)


def main():
    args = parser.parse_args()
    print(f'Model dir: {args.model_dir}')

    if args.visualize:
        args.test = False

    if args.test or args.test_all or args.visualize or args.fast_dev_run:
        args.slurm = False
        args.n_gpus = 1
        args.num_node = 1
        args.progress_bar = True
        args.batch_size = 16

    if args.test or args.test_all or args.visualize:
        args.batch_size = 12 if args.model == 'wavenet' else (4 if args.model == 'generative_wavenet' else 32)
        args.num_worker = 10

    if args.visualize:
        args.audio_len = 40960

    if args.fast_dev_run:
        if os.path.exists('data/models/debug'):
            shutil.rmtree('data/models/debug')
        args.model_dir = 'data/models/debug'
        args.version = 'v1'
        args.batch_size = 2

    folder = args.model_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    if args.slurm:
        import submitit
        executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
        executor.update_parameters(slurm_job_name=args.version, timeout_min=60*48, slurm_partition=args.part,
                                   nodes=args.num_node, gpus_per_node=args.n_gpus, cpus_per_task=10,
                                   slurm_constraint='volta32gb' if args.gpu_mem32 else None, slurm_mem=100 * 1024,
                                   tasks_per_node=args.n_gpus, comment=args.comment
                                   )
        job = executor.submit(run, args)
        print(job.job_id)
    else:
        run(args)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.dataset == 'replay_nvas':
        from nvas.datasets.replay_nvas_dataset import ReplayNVASDataset
        dataset = ReplayNVASDataset
    elif args.dataset == 'soundspaces_nvas':
        from nvas.datasets.soundspaces_nvas_dataset import SoundSpacesNVASDataset
        dataset = SoundSpacesNVASDataset
    else:
        raise ValueError

    if args.test or args.test_all or args.visualize:
        test_set = dataset(split=args.test_split, args=args)
        test_dataset = torch.utils.data.DataLoader(test_set, num_workers=1, batch_size=args.batch_size,
                                                   pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker)
    else:
        train_set = dataset(split='train', args=args)
        train_dataset = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=args.num_worker,
                                                    pin_memory=True, batch_size=args.batch_size, persistent_workers=True)
        val_set = dataset(split='val', args=args)
        val_dataset = torch.utils.data.DataLoader(val_set, num_workers=args.num_worker, batch_size=args.batch_size,
                                                  pin_memory=True, persistent_workers=True)

    if args.model == 'vigas':
        from nvas.models.vigas import ViGAS
        model = ViGAS(args)
    else:
        raise ValueError
    # Model training
    logger = loggers.TensorBoardLogger(
        args.model_dir,
        version=args.version
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.model_dir, args.version),
        filename="{epoch:04d}",
        every_n_epochs=args.ckpt_interval,
        save_top_k=-1,
        verbose=True,
    )

    if args.eval_last or (args.auto_resume and not args.test):
        existing_checkpoints = sorted(glob.glob(os.path.join(args.model_dir, args.version, f'epoch=*.ckpt')))
        if len(existing_checkpoints) != 0:
            args.from_pretrained = existing_checkpoints[-1]
            print(args.from_pretrained)
        else:
            print('There is no existing checkpoint!')

    if args.eval_ckpt != -1:
        args.from_pretrained = os.path.join(args.model_dir, args.version, f'epoch={args.eval_ckpt:04}.ckpt')
        print(args.from_pretrained)

    if args.eval_best:
        args.from_pretrained = os.path.join(args.model_dir, args.version, f'best_val.ckpt')
        print(args.from_pretrained)

    progress_callback = TQDMProgressBar(refresh_rate=args.fast_dev_run or args.progress_bar)

    trainer = Trainer(
        gpus=args.n_gpus,
        num_nodes=args.num_node,
        # auto_select_gpus=True,
        strategy="ddp",
        # auto_scale_batch_size="binsearch",
        benchmark=True,
        max_epochs=args.max_epochs,
        default_root_dir=args.model_dir,
        callbacks=[checkpoint_callback, progress_callback],
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="value"
        # profiler='simple',
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches=1,
        # num_sanity_val_steps=0,
        # max_epochs=1,
        # profiler='simple'
        # overfit_batches=10
    )

    if not args.test and not args.test_all and not args.visualize:
        trainer.fit(model, train_dataset, val_dataset, ckpt_path=args.from_pretrained)
    elif args.test:
        if model.learnable:
            model.load_weights(torch.load(args.from_pretrained, map_location='cpu'))
        trainer.test(model, dataloaders=test_dataset)
        # model.save_test_stats()
    elif args.test_all:
        args.eval_best = False
        ckpt_paths = sorted(glob.glob(os.path.join(args.model_dir, args.version, f'epoch=*.ckpt')))
        for ckpt_path in ckpt_paths:
            args.from_pretrained = ckpt_path
            print(ckpt_path)
            tokens = args.from_pretrained.split('=')
            ckpt = int(tokens[-1][:tokens[-1].find('.')])
            if ckpt <= args.prev_ckpt:
                continue
            model.load_weights(torch.load(ckpt_path, map_location='cpu'))
            trainer.test(model, dataloaders=test_dataset)
    elif args.visualize:
        if model.learnable:
            model.load_weights(torch.load(args.from_pretrained, map_location='cpu'))
        model.to(device=torch.device('cuda'))
        model.eval()
        with torch.no_grad():
            generate_qual(args, model, test_dataset)
    else:
        raise ValueError


def generate_qual(args, model, dataset):
    import json
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from librosa.display import waveshow

    qual_dir = os.path.join(args.model_dir, args.version, "qual")
    os.makedirs(qual_dir, exist_ok=True)
    count = 0
    for i, batch in enumerate(dataset):
        output = model.audio_synthesis(batch, i, phase='test')
        pred, tgt = output['pred'], output['tgt']

        wav_dict = {
            'pred': pred if args.decode_wav else model.spec2wav(pred, batch['src_wav']),
            'tgt': batch['tgt_wav'],
            'input': batch['src_wav'] if not args.dereverb_input else batch['src_wav_copy']
        }
        spec_dict = {}
        for key in wav_dict:
            wav = wav_dict[key].reshape(-1, wav_dict[key].shape[-1])
            spec = torch.log1p(torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                               window=torch.hamming_window(400, device=wav_dict[key].device), pad_mode='constant',
                               center=True, return_complex=True).abs())
            spec = spec[:, :spec.shape[1]//2, :]
            spec_dict[key] = spec.reshape(pred.shape[0], -1, *spec.shape[1:])

        for dic in [wav_dict, spec_dict]:
            for key, val in dic.items():
                dic[key] = val.cpu().numpy()

        for j in tqdm(range(pred.shape[0])):
            y_limit = max([np.abs(wav[j]).max() for wav in wav_dict.values()]) + 0.03
            audio_duration = wav_dict['input'].shape[-1] / args.sample_rate
            step = 0.2 if args.audio_len == 16000 else 0.5
            x_ticks = np.arange(step, audio_duration, step=step)

            for k in range(2):
                fig, axes = plt.subplots(2, 3)
                axes[0][0].imshow(spec_dict['input'][j][k])
                axes[0][0].set_aspect(1 / axes[0][0].get_data_ratio())
                axes[0][0].invert_yaxis()
                axes[0][0].set_title('Input')

                plt.sca(axes[1][0])
                waveshow(wav_dict['input'][j][k], args.sample_rate)
                axes[1][0].set_ylim(-y_limit, y_limit)
                axes[1][0].set_xticks(x_ticks)

                axes[0][1].imshow(spec_dict['pred'][j][k])
                axes[0][1].set_aspect(1 / axes[0][1].get_data_ratio())
                axes[0][1].invert_yaxis()
                axes[0][1].set_title('Prediction')

                plt.sca(axes[1][1])
                waveshow(wav_dict['pred'][j][k], args.sample_rate)
                axes[1][1].set_ylim(-y_limit, y_limit)
                axes[1][1].set_xticks(x_ticks)

                axes[0][2].imshow(spec_dict['tgt'][j][k])
                axes[0][2].set_aspect(1 / axes[0][2].get_data_ratio())
                axes[0][2].invert_yaxis()
                axes[0][2].set_title('Target')

                plt.sca(axes[1][2])
                waveshow(wav_dict['tgt'][j][k], args.sample_rate)
                axes[1][2].set_ylim(-y_limit, y_limit)
                axes[1][2].set_xticks(x_ticks)

                fig.tight_layout()
                plt.savefig(os.path.join(qual_dir, f"{count}-plot-{'left' if k == 0 else 'right'}.png"))
                plt.close(fig)

            if 'src_rgb' in batch:
                rgb = batch['src_rgb_copy'] if 'src_rgb_copy' in batch else batch['src_rgb']
                rgb = rgb.cpu()[j].permute(1, 2, 0).clip(0, 1).numpy()  # L/R channels are displayed separately
                plt.imsave(os.path.join(qual_dir, f'{count}-src-rgb.png'), rgb)
            if 'tgt_rgb' in batch:
                rgb = batch['tgt_rgb_copy'] if 'tgt_rgb_copy' in batch else batch['tgt_rgb']
                rgb = rgb.cpu()[j].permute(1, 2, 0).clip(0, 1).numpy()
                if args.use_tgt_bboxes_as_mask:
                    plt.imsave(os.path.join(qual_dir, f'{count}-tgt-rgb-mask.png'), rgb[:, :, 3])
                    rgb = rgb[:, :, :3]
                plt.imsave(os.path.join(qual_dir, f'{count}-tgt-rgb.png'), rgb)

            wavfile.write(os.path.join(qual_dir, f'{count}-input.wav'), args.sample_rate, wav_dict['input'][j].T)
            wavfile.write(os.path.join(qual_dir, f'{count}-tgt.wav'), args.sample_rate, wav_dict['tgt'][j].T)
            wavfile.write(os.path.join(qual_dir, f'{count}-pred.wav'), args.sample_rate, wav_dict['pred'][j].T)
            stats = {}
            for key_word in ['tgt_pose', 'tgt_rotation_degree', 'speaker_pose']:
                if key_word in batch:
                    stats[key_word] = batch[key_word].cpu()[j].numpy().tolist()
            with open(os.path.join(qual_dir, f'{count}-stats.pkl'), 'wb') as fo:
                pickle.dump(stats, fo)
            count += 1
            
        if count > 31:
            break


if __name__ == "__main__":
    main()
