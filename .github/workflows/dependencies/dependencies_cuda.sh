#!/usr/bin/env bash

set -eu -o pipefail

# Install CUDA 12 toolkit on Ubuntu 22.04

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    cuda-compiler-12-4 \
    cuda-minimal-build-12-4 \
    cuda-cudart-dev-12-4

sudo ln -sfn /usr/local/cuda-12.4 /usr/local/cuda
