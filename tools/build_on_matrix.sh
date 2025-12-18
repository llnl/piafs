#!/bin/bash
# Script to build PIAFS on matrix.llnl.gov with GPU support
# Usage: ./scripts/build_on_matrix.sh

set -e  # Exit on error

REMOTE_HOST="matrix.llnl.gov"
REMOTE_USER="ghosh5"
REMOTE_DIR="/g/g92/ghosh5/Codes/piafs"
BUILD_DIR="build.matrix"

echo "=========================================="
echo "Building PIAFS on ${REMOTE_HOST}"
echo "=========================================="

# Use bash -l to get login shell and source .bashrc for conditional module loading
ssh ${REMOTE_USER}@${REMOTE_HOST} 'bash -l -c "
set -e

# The .bashrc on LC machines has conditionals based on LC_HOST
# For non-interactive shells, bash -l sources .bash_profile/.profile, not .bashrc
# So we need to explicitly source .bashrc
# Also check .bash_profile in case modules are loaded there
if [ -f ~/.bash_profile ]; then
    echo \"Sourcing ~/.bash_profile...\"
    set +e
    . ~/.bash_profile 2>/dev/null || true
    set -e
fi

# Initialize module system first
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash 2>/dev/null || true
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

# Source .bashrc to load modules (handles conditional loading based on LC_HOST)
if [ -f ~/.bashrc ]; then
    echo \"Sourcing ~/.bashrc to load modules...\"
    # Temporarily disable exit on error and source .bashrc
    set +e
    # Source in current shell (not subshell) so environment persists
    # Some .bashrc files check if interactive and return early, so we bypass that
    BASH_ENV=~/.bashrc bash -c \"\" 2>/dev/null || true
    . ~/.bashrc 2>/dev/null || true
    set -e
fi

cd /g/g92/ghosh5/Codes/piafs

echo \"Pulling latest changes...\"
git pull

echo \"Checking for CUDA compiler...\"
if ! command -v nvcc >/dev/null 2>&1; then
    echo \"Warning: CUDA compiler (nvcc) not found after loading modules from .bashrc\"
    echo \"Please ensure module load commands are in ~/.bashrc\"
fi
echo \"CUDA compiler: $(which nvcc 2>/dev/null || echo not found)\"

echo \"Navigating to build directory...\"
cd build.matrix

echo \"Cleaning build directory...\"
rm -rf *

echo \"Configuring CMake...\"
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -S .. -B .

echo \"Building...\"
make -j 36

echo \"Installing...\"
make install

echo \"==========================================\"
echo \"Build completed successfully!\"
echo \"==========================================\"
"'

echo "Remote build completed!"

