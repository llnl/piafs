#!/usr/bin/env bash

set -eu -o pipefail

# Install ROCm 6.x / HIP on Ubuntu 22.04

wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
    | sudo gpg --dearmor -o /usr/share/keyrings/rocm-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] \
https://repo.radeon.com/rocm/apt/6.0/ jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    hip-dev \
    rocm-cmake
