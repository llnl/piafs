# Building PIAFS with GPU Support

This document describes how to build PIAFS with GPU support using CUDA or HIP.

## Prerequisites

### For CUDA:
- CUDA Toolkit (version 11.0 or later recommended)
- NVIDIA GPU with compute capability 7.5 or higher (default: sm_75)
- CMake 3.10 or later

### For HIP:
- ROCm (version 4.0 or later recommended)
- AMD GPU with ROCm support
- CMake 3.10 or later

## Building with CUDA

```bash
mkdir build
cd build
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON ..
make -j 4
```

### Advanced CUDA Options:

```bash
# Specify CUDA architecture (default is 75 if not specified)
# Use comma-separated list for multiple architectures
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      ..

# Build for multiple architectures
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
      ..

# Use specific CUDA compiler
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      ..
```

## Building with HIP

```bash
mkdir build
cd build
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON ..
make -j 4
```

### Advanced HIP Options:

```bash
# Specify HIP compiler
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      ..
```

## Building without GPU (CPU only)

```bash
mkdir build
cd build
cmake ..
make -j 4
```

## Combined Options

You can combine GPU support with other options:

```bash
# GPU + MPI + OpenMP
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON \
      -DENABLE_OMP=ON \
      ..

# Serial GPU build (no MPI)
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON \
      -DENABLE_SERIAL=ON \
      ..
```

## Verification

After building, verify GPU support:

```bash
# Check if GPU library was built
ls build/src/GPU/libGPU.a

# Run a test (if available)
# Note: Binary name includes version and build info (e.g., PIAFS-v0.1-gcc9-mpi4)
./build/src/PIAFS-v0.1-gcc9-mpi4 --help
```

## Troubleshooting

### CUDA not found
- Ensure CUDA is installed and `nvcc` is in PATH
- Set `CUDA_PATH` environment variable
- Use `-DCMAKE_CUDA_COMPILER` to specify nvcc path

### HIP not found
- Ensure ROCm is installed and `hipcc` is in PATH
- Set `ROCM_PATH` environment variable
- Use `-DCMAKE_CXX_COMPILER` to specify hipcc path

### Compilation errors
- Check GPU compute capability compatibility
- Verify CUDA/HIP version compatibility
- Check compiler flags in CMake output

## Performance Notes

- GPU support requires data to be on device
- First run may be slower due to kernel compilation
- Ensure sufficient GPU memory for your problem size
- Use `nvidia-smi` (CUDA) or `rocm-smi` (HIP) to monitor GPU usage

