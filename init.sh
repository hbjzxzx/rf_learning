#! /bin/bash

conda init
source ~/.bashrc
conda create rl
conda install swit
conda activate rl
conda install --file requirements.txt

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia