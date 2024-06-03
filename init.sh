#! /bin/bash

conda init
source ~/.bashrc
yes | conda install swig
pip install -r requirements.txt 

# Install PyTorch

if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is installed"
    # Install PyTorch for GPU
    yes | conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia
else
    echo "nvcc is not installed"
    # Install PyTorch for CPU
    yes | conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
fi