#! /bin/bash

conda init
source ~/.bashrc
conda create rl
conda install swit
conda activate rl
conda install --file requirements.txt