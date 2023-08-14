#!/bin/bash

mkdir data && cd data

# download the synthetic data
mkdir synthetic_dataset && cd synthetic_dataset
wget https://dl.fbaipublicfiles.com/large_objects/nvas/v16.zip
unzip v16.zip
cd ..

# download the librispeech data
wget https://dl.fbaipublicfiles.com/large_objects/nvas/LibriSpeech-wav.zip
unzip LibriSpeech-wav.zip

# download the real data
mkdir appen_dataset && cd appen_dataset
wget https://dl.fbaipublicfiles.com/large_objects/nvas/v3.zip
unzip v3.zip
cp ../../res/camera_positions_fixed_rotation.json .
cd ..

# donwloand the pretrained models 
wget https://dl.fbaipublicfiles.com/nvas/models.zip
unzip models.zip