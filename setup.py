# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


setup(
    name='novel-view-acoustic-synthesis',
    version='0.1.1',
    packages=[
        'nvas'
    ],
    install_requires=[
        'pytorch_lightning==1.9.3',
        'torch==1.13.1',
        'numpy>=1.16.1',
        'opencv-python>=3.3.0',
        'imageio>=2.2.0',
        'imageio-ffmpeg>=0.2.0',
        'scipy>=1.0.0',
        'tqdm>=4.0.0',
        'getch',
        'matplotlib',
        'librosa',
        'torchsummary',
        'tqdm',
        'scikit-image',
        'torchaudio',
        'torchvision',
        'decord',
        'albumentations',
        'submitit',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)