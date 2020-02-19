#!/bin/sh

# activate conda environment
conda activate infomax

git clone https://github.com/NVIDIA/apex
cd apex

# set CUDA_HOME to CUDA 10.1, since PyTorch's binaries are compiled against 10.1
CUDA_HOME=/usr/local/cuda-10.1 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./