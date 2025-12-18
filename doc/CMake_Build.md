# Building PIAFS with CMake {#cmake_build}

[TOC]

## Overview

PIAFS supports CMake as a modern build system alternative to autotools. CMake provides:
- Better IDE integration (Visual Studio Code, CLion, Eclipse)
- Faster configuration
- Improved cross-platform support
- Native out-of-tree builds

## Quick Start

### Default Build (with MPI)
```bash
mkdir build && cd build
cmake ..
make -j4
```
Executable: `build/src/PIAFS-<compiler>-<mpi/serial>` (e.g., `PIAFS-gcc-mpi`)

The binary name automatically includes compiler and build configuration information to distinguish different builds.

### Common Build Variants
```bash
# Serial (no MPI)
cmake -DENABLE_SERIAL=ON ..

# With OpenMP
cmake -DENABLE_OMP=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# With CUDA GPU support (NVIDIA)
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..

# With HIP GPU support (AMD)
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 ..
```

## Build Options Summary

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_SERIAL` | OFF | Build without MPI |
| `ENABLE_OMP` | OFF | Enable OpenMP |
| `ENABLE_GPU` | OFF | Enable GPU support (requires CUDA or HIP) |
| `ENABLE_CUDA` | OFF | Enable CUDA (NVIDIA GPUs) |
| `ENABLE_HIP` | OFF | Enable HIP (AMD GPUs) |
| `ENABLE_GPU_AWARE_MPI` | OFF | Enable GPU-aware MPI (**⚠️ Not currently working - do not use**) |
| `CMAKE_CUDA_ARCHITECTURES` | Auto-detected | CUDA compute capability (e.g., 90 for H100, 80 for A100) |
| `CMAKE_HIP_ARCHITECTURES` | Not set | HIP GPU architecture (e.g., gfx942 for MI300A) |
| `CMAKE_BUILD_TYPE` | Release | Debug, Release, RelWithDebInfo, MinSizeRel |
| `CMAKE_INSTALL_PREFIX` | Source directory | Installation directory (matches autotools) |
| `MPIEXEC` | Auto-detected | MPI run command for tests (e.g., mpiexec, srun) |

## GPU Support

PIAFS supports GPU acceleration through both CUDA (NVIDIA) and HIP (AMD) backends.

### CUDA (NVIDIA GPUs)

For systems with NVIDIA GPUs, enable CUDA support:

```bash
mkdir build && cd build
cmake .. \
  -DENABLE_GPU=ON \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
make -j8
make install
```

**Common CUDA Architectures:**
- `90` - NVIDIA H100 (Hopper)
- `80` - NVIDIA A100 (Ampere)
- `75` - NVIDIA RTX 20xx/Titan RTX/T4 (Turing)
- `70` - NVIDIA V100 (Volta)

To find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

### HIP (AMD GPUs)

For systems with AMD GPUs, enable HIP support:

```bash
mkdir build && cd build
cmake .. \
  -DENABLE_GPU=ON \
  -DENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
make -j8
make install
```

**Important for HIP builds:**
- When using MPI wrappers configured with `rocmcc`/`hipcc`, CMake will automatically detect them
- For standalone builds, set `CMAKE_CXX_COMPILER=hipcc`
- The GPU architecture (`CMAKE_HIP_ARCHITECTURES`) must be specified

**Common AMD GPU Architectures:**
- `gfx942` - AMD MI300A/MI300X (CDNA 3)
- `gfx90a` - AMD MI250X/MI250 (CDNA 2)
- `gfx908` - AMD MI100 (CDNA 1)
- `gfx906` - AMD MI50/MI60 (GCN 5)

To find your GPU architecture:
```bash
rocminfo | grep "Name:" | grep gfx
```

### Running with GPUs

GPU execution is controlled at runtime via environment variables:

```bash
# Enable GPU acceleration
export PIAFS_USE_GPU=1

# Run with MPI and GPUs
mpirun -np 4 bin/piafs <input_file>

# Or with Slurm
srun -n 4 --gpus-per-node=4 bin/piafs <input_file>
```

**GPU Environment Variables:**
- `PIAFS_USE_GPU=1` - Enable GPU acceleration (default: 0/disabled)
- `PIAFS_GPU_VERBOSE=1` - Enable verbose GPU output
- `PIAFS_GPU_VALIDATE=1` - Enable GPU result validation
- `PIAFS_GPU_SYNC_EVERY_OP=1` - Force synchronization after every operation (debugging)

**Important Notes:**
- GPU support must be enabled at compile time (`-DENABLE_GPU=ON`)
- Each MPI rank is automatically assigned to a GPU based on local rank
- If there are more MPI ranks than GPUs on a node, ranks will share GPUs (with a warning)
- All computations and memory allocations occur on the GPU
- Data is only copied to the host for I/O operations

### GPU-Aware MPI

**⚠️ NOTE: GPU-aware MPI is currently not working and is disabled by default. Use the standard (non-GPU-aware) MPI path which copies data through host memory. This has minimal performance impact for typical problem sizes.**

GPU-aware MPI allows MPI to transfer data directly between GPU memory across nodes without staging through host memory. This can significantly improve communication performance for multi-GPU runs, but requires specific MPI library support.

#### Requirements

GPU-aware MPI requires an MPI library built with GPU support:

**For CUDA (NVIDIA GPUs):**
- OpenMPI built with `--with-cuda`
- MVAPICH2-GDR
- Cray MPICH with GPU support
- NVIDIA HPC-X

**For HIP (AMD GPUs):**
- OpenMPI built with ROCm support
- Cray MPICH with ROCm support

#### Enabling GPU-Aware MPI

```bash
# CUDA with GPU-aware MPI
cmake .. \
  -DENABLE_GPU=ON \
  -DENABLE_CUDA=ON \
  -DENABLE_GPU_AWARE_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90

# HIP with GPU-aware MPI
cmake .. \
  -DENABLE_GPU=ON \
  -DENABLE_HIP=ON \
  -DENABLE_GPU_AWARE_MPI=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
```

#### How It Works

Without GPU-aware MPI, halo exchange involves:
1. Pack halo data on GPU
2. Copy packed data from GPU to host buffer
3. MPI sends/receives using host buffers
4. Copy received data from host to GPU buffer
5. Unpack halo data on GPU

With GPU-aware MPI enabled:
1. Pack halo data directly to GPU send buffer
2. MPI sends/receives using GPU buffers directly
3. Unpack halo data directly from GPU receive buffer

This eliminates two host-device memory copies per halo exchange, which can significantly reduce communication overhead.

#### Verifying GPU-Aware MPI Support

**OpenMPI:**
```bash
# Check if OpenMPI was built with CUDA support
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
# Should return: mpi_built_with_cuda_support:value:true

# For ROCm/HIP support
ompi_info --parsable --all | grep accelerator
```

**MVAPICH2:**
```bash
# Check for GDR (GPUDirect RDMA) support
mpirun --version | grep -i cuda
```

**Environment Variables (OpenMPI):**
```bash
# May be needed to enable GPU-aware features
export OMPI_MCA_opal_cuda_support=1
```

#### Performance Considerations

- GPU-aware MPI is most beneficial for large message sizes and frequent communication
- For small grids or infrequent communication, the overhead may not be significant
- GPUDirect RDMA (peer-to-peer GPU communication) provides additional speedup on supported systems
- Network topology and GPU placement affect performance

#### Troubleshooting

**Segmentation fault with GPU-aware MPI:**
- Verify your MPI library actually supports GPU-aware communication
- Check that GPU buffers are properly allocated (not host memory)
- Ensure `PIAFS_USE_GPU=1` is set at runtime

**No performance improvement:**
- GPU-aware MPI benefits depend on problem size and communication patterns
- Small problems may not see significant speedup
- Profile with NVIDIA Nsight Systems or ROCm profiler to identify bottlenecks

## Installation

By default, CMake installs to the project's own directory (matching autotools behavior):

```bash
make install
```

This creates:
- `bin/PIAFS-<compiler>-<mpi/serial>` - Executable

The binary name format is: `PIAFS-<compiler>-<mpi/serial>[-omp][-<buildtype>]`

Examples:
- `PIAFS-gcc-mpi` - GCC compiler, MPI parallel, Release build
- `PIAFS-clang-mpi-omp` - Clang compiler, MPI parallel, OpenMP enabled
- `PIAFS-gcc-mpi-debug` - GCC compiler, MPI parallel, Debug build
- `PIAFS-intel-serial` - Intel compiler, serial mode

**Note:** The default install prefix is the source directory, so no root permissions are needed.

To install to a different location:
```bash
cmake -DCMAKE_INSTALL_PREFIX=/your/path ..
make install
```

**Important:** If installing to system directories like `/usr/local`, you may need root permissions:
```bash
sudo make install
```

Alternatively, install to a user-writable location:
```bash
cmake -DCMAKE_INSTALL_PREFIX=$HOME/piafs ..
make install
```

## Testing

PIAFS includes a comprehensive regression test suite that compares simulation outputs against benchmark solutions.

### Running Tests

```bash
# Run all tests
make test

# Or use ctest directly
ctest

# Verbose output
ctest --verbose

# Show output only for failed tests
ctest --output-on-failure
```

### Custom MPI Executor (HPC Platforms)

For HPC systems with job schedulers, specify a custom MPI launch command:

```bash
# For Slurm
cmake -DMPIEXEC="srun" ..

# For IBM JSRun
cmake -DMPIEXEC="jsrun" ..

# For standard mpirun with options
cmake -DMPIEXEC="mpirun -np" ..
```

The test suite will use the specified command to launch parallel tests.

## Machine-Specific Build Instructions (LLNL)

### Matrix (NVIDIA H100 GPUs - CUDA)

Matrix has NVIDIA H100 GPUs (compute capability 9.0).

**Build:**
```bash
cd ~/piafs
rm -rf build
mkdir build
cd build

cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..

make -j8
make install
```

**Run:**
```bash
# Interactive session
salloc -N 1 --gpus-per-node=4 -p pbatch -t 60

# Set environment
export PIAFS_USE_GPU=1

# Run with 4 MPI ranks, 4 GPUs
srun -n 4 --gpus-per-node=4 bin/piafs <input_file>
```

**Common CUDA Architectures:**
- `90` - NVIDIA H100 (Matrix)
- `80` - NVIDIA A100
- `75` - NVIDIA V100
- `70` - NVIDIA V100 (older)

### Tuolumne (AMD MI300A GPUs - HIP)

Tuolumne has AMD MI300A GPUs (architecture gfx942).

**Build:**
```bash
cd ~/piafs
rm -rf build
mkdir build
cd build

# The MPI wrapper is already configured with rocmcc
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 ..

make -j8
make install
```

**Run:**
```bash
# Interactive session
salloc -N 1 --gpus-per-node=4 -p mi300a -t 60

# Set environment
export PIAFS_USE_GPU=1

# Run with 4 MPI ranks, 4 GPUs
srun -n 4 --gpus-per-node=4 bin/piafs <input_file>
```

**Common AMD GPU Architectures:**
- `gfx942` - AMD MI300A (Tuolumne)
- `gfx90a` - AMD MI250X
- `gfx908` - AMD MI100
- `gfx906` - AMD MI50/MI60

### Dane (CPU Only)

Dane is a CPU-only cluster (Intel Xeon processors).

**Build:**
```bash
cd ~/piafs
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release

# Optional: Enable OpenMP for shared-memory parallelism
# cmake -DENABLE_OMP=ON -DCMAKE_BUILD_TYPE=Release ..

make -j8
make install
```

**Run:**
```bash
# Interactive session
salloc -N 2 -p pbatch -t 60

# Run with 128 MPI ranks (64 per node)
srun -N 2 -n 128 bin/piafs <input_file>
```

**With OpenMP (hybrid MPI+OpenMP):**
```bash
# 4 MPI ranks per node, 16 OpenMP threads per rank
export OMP_NUM_THREADS=16
srun -N 2 -n 8 -c 16 bin/piafs <input_file>
```

### Test Output

Test results are logged and can be reviewed:
- CTest output shows pass/fail status
- Detailed logs available in the test working directory
- Benchmark comparisons use relative tolerance of 1.0e-14

For more details, see `Tests/README.md` in the source directory.

## Detailed Documentation

For comprehensive build instructions, advanced options, IDE integration, and troubleshooting:
- See **BUILD_CMAKE.md** in the project root directory for complete CMake documentation
- See **CMAKE_MIGRATION.md** for migration guide from autotools

## Quick Reference: CMake vs Autotools

| Task | Autotools | CMake |
|------|-----------|-------|
| Configure | `./configure` | `cmake ..` |
| Serial | `--enable-serial` | `-DENABLE_SERIAL=ON` |
| OpenMP | `--enable-omp` | `-DENABLE_OMP=ON` |
| CUDA GPU | N/A | `-DENABLE_GPU=ON -DENABLE_CUDA=ON` |
| HIP GPU | N/A | `-DENABLE_GPU=ON -DENABLE_HIP=ON` |
| GPU-aware MPI | N/A | `-DENABLE_GPU_AWARE_MPI=ON` |
| MPI executor | `--with-mpiexec=srun` | `-DMPIEXEC=srun` |
| Build | `make` | `make` |
| Test | `make check` | `make test` or `ctest` |
| Clean | `make clean` | `rm -rf build` |

## Troubleshooting

**MPI not found:**
```bash
cmake -DMPI_C_COMPILER=/path/to/mpicc -DMPI_CXX_COMPILER=/path/to/mpicxx ..
```

**Build from scratch:**
```bash
rm -rf build && mkdir build && cd build && cmake .. && make
```

## Startup Information

When PIAFS starts, it displays build configuration and system information:

**Serial mode:**
```
================================================================================
PIAFS - Serial Version
  Version: 0.1
  Git Hash: abc1234 (branch: main)
  Build Date: 2024-01-15 10:30:45
  Build Type: Release
  MPI Mode: serial
  OpenMP: enabled
  GPU Devices: 4
================================================================================
```

**MPI parallel mode:**
```
================================================================================
PIAFS - Parallel (MPI) version with 64 processes
  Version: 0.1
  Git Hash: abc1234 (branch: main)
  Build Date: 2024-01-15 10:30:45
  Build Type: Release
  OpenMP: enabled
  GPU Devices: 16 total (4 per node, 4 nodes) (GPU ENABLED)
================================================================================
```

This information helps identify:
- Execution mode (serial vs MPI parallel)
- Number of MPI processes
- Version and git commit for reproducibility
- Build configuration (OpenMP, GPU support)
- GPU device count and status (if GPU-enabled build)

**GPU Status Messages:**
- `(GPU ENABLED)` - GPU acceleration is active
- `(GPU DISABLED - set PIAFS_USE_GPU=1 to enable)` - GPU support compiled but not enabled at runtime
