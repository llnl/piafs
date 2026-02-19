#!/usr/bin/env bash

set -eu -o pipefail

# Install ROCm 6.x / HIP on Ubuntu 22.04
# Based on: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html

# Make the directory if it doesn't exist yet
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# Download and setup GPG key
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

source /etc/os-release # set UBUNTU_CODENAME

VERSION=${1:-6.0}

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/${VERSION}/ ${UBUNTU_CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

# Setup ROCm PATH environment
echo 'export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/opt/rocm/hip/bin:$PATH' \
    | sudo tee -a /etc/profile.d/rocm.sh

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    rocm-dev${VERSION} \
    rocm-hip-runtime${VERSION}

# Activate ROCm environment
source /etc/profile.d/rocm.sh
hipcc --version
