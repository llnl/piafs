# Building PIAFS

PIAFS supports two build systems: **CMake** (recommended) and **Autotools** (traditional).

```{contents}
:local:
:depth: 2
```

## Build System Comparison

| Feature | CMake | Autotools |
|---------|-------|-----------|
| GPU Support (CUDA/HIP) | **Yes** | No |
| IDE Integration | Excellent | Limited |
| Configuration | `cmake ..` | `./configure` |
| Testing | `make test` / `ctest` | `make check` |
| Cross-platform | All platforms | Unix-like only |

**Recommendation:** Use CMake, especially if you need GPU acceleration.

## CMake Build (Recommended)

### Quick Start

Basic build with MPI:

```bash
mkdir build
cd build
cmake ..
make -j 4
make install
```

The executable will be at `bin/PIAFS-<compiler>-mpi` (e.g., `PIAFS-gcc-mpi`).

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_SERIAL` | OFF | Build without MPI |
| `ENABLE_OMP` | OFF | Enable OpenMP |
| `ENABLE_GPU` | OFF | Enable GPU support |
| `ENABLE_CUDA` | OFF | Enable CUDA (NVIDIA) |
| `ENABLE_HIP` | OFF | Enable HIP (AMD) |
| `CMAKE_CUDA_ARCHITECTURES` | Auto | CUDA compute capability |
| `CMAKE_HIP_ARCHITECTURES` | - | HIP GPU architecture |
| `CMAKE_BUILD_TYPE` | Release | Build type |
| `MPIEXEC` | Auto | MPI run command |

### Common Build Configurations

**Serial build (no MPI):**

```bash
cmake -DENABLE_SERIAL=ON ..
make -j 4
make install
```

**With OpenMP:**

```bash
cmake -DENABLE_OMP=ON ..
make -j 4
make install
```

**With CUDA GPU support (NVIDIA):**

```bash
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..
make -j 4
make install
```

Common CUDA architectures:
- `90` - H100 (Hopper)
- `80` - A100 (Ampere)
- `75` - RTX 20xx/Titan RTX (Turing)
- `70` - V100 (Volta)

Find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**With HIP GPU support (AMD):**

```bash
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 ..
make -j 4
make install
```

Common AMD GPU architectures:
- `gfx942` - MI300A/MI300X
- `gfx90a` - MI250X/MI250
- `gfx908` - MI100
- `gfx906` - MI50/MI60

Find your GPU architecture:
```bash
rocminfo | grep "Name:" | grep gfx
```

**Debug build:**

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j 4
make install
```

### Running Tests (CMake)

```bash
cd build
make test

# Or with ctest for more control
ctest --verbose
ctest --output-on-failure
```

For HPC systems with job schedulers:

```bash
cmake -DMPIEXEC="srun" ..
make test
```

### GPU Execution

GPU execution is **enabled by default** for GPU builds. Use environment variables to control behavior:

```bash
# Run with GPUs (default for GPU builds)
mpirun -np 4 bin/PIAFS-gcc-mpi

# Or with Slurm
srun -n 4 --gpus-per-node=4 bin/PIAFS-gcc-mpi

# To disable GPU and run on CPU (if needed)
export PIAFS_USE_GPU=0
mpirun -np 4 bin/PIAFS-gcc-mpi
```

GPU environment variables:
- `PIAFS_USE_GPU=0` - Disable GPU acceleration (default is 1 for GPU builds)
- `PIAFS_GPU_VERBOSE=1` - Enable verbose GPU output
- `PIAFS_GPU_VALIDATE=1` - Enable GPU result validation
- `PIAFS_GPU_SYNC_EVERY_OP=1` - Force synchronization (debugging)

## Autotools Build

```{warning}
Autotools does **not** support GPU acceleration. Use CMake for GPU builds.
```

### Quick Start

```bash
autoreconf -i
./configure
make -j 4
make install
```

The executable will be at `bin/PIAFS-<compiler>-mpi`.

**Note:** `autoreconf -i` is only needed:
- After a fresh clone/download
- When source files are added/removed
- After major structural changes

### Configure Options

**Serial build:**

```bash
./configure --enable-serial
make -j 4
make install
```

**With OpenMP:**

```bash
./configure --enable-omp
make -j 4
make install
```

**Custom MPI directory:**

```bash
./configure --with-mpi-dir=/path/to/mpi
make -j 4
make install
```

**Custom install location:**

```bash
./configure --prefix=/path/to/install
make -j 4
make install
```

**Debug build:**

```bash
CFLAGS="-g -O0" CXXFLAGS="-g -O0" ./configure
make -j 4
make install
```

### Running Tests (Autotools)

```bash
make check
```

For HPC systems:

```bash
./configure --with-mpiexec="srun"
make check
```

### Cleaning Up

```bash
# Clean build artifacts
make clean

# Remove all generated files (requires reconfigure)
make distclean

# Complete cleanup (requires autoreconf -i again)
make maintainer-clean
```

## Binary Naming Convention

PIAFS binaries are named to identify their configuration:

Format: `PIAFS-<compiler>-<mode>[-omp][-debug]`

Examples:
- `PIAFS-gcc-mpi` - GCC, MPI parallel, Release
- `PIAFS-clang-mpi-omp` - Clang, MPI, OpenMP enabled
- `PIAFS-gcc-mpi-debug` - GCC, MPI, Debug build
- `PIAFS-intel-serial` - Intel compiler, serial mode

## Installation

By default, PIAFS installs to `bin/` in the source directory. This requires no special permissions.

**CMake custom install:**

```bash
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make install
```

**Autotools custom install:**

```bash
./configure --prefix=/path/to/install
make install
```

## Machine-Specific Examples (LLNL HPC)

### Matrix (NVIDIA H100 GPUs)

```bash
mkdir build && cd build
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..
make -j8 && make install
```

Run:
```bash
salloc -N 1 --gpus-per-node=4 -p pbatch -t 60
# GPU enabled by default for GPU builds
srun -n 4 --gpus-per-node=4 bin/PIAFS-gcc-mpi
```

### Tuolumne (AMD MI300A GPUs)

```bash
mkdir build && cd build
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 ..
make -j8 && make install
```

Run:
```bash
salloc -N 1 --gpus-per-node=4 -p mi300a -t 60
# GPU enabled by default for GPU builds
srun -n 4 --gpus-per-node=4 bin/PIAFS-gcc-mpi
```

### Dane (CPU Only)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 && make install
```

Run:
```bash
salloc -N 2 -p pbatch -t 60
srun -N 2 -n 128 bin/PIAFS-gcc-mpi
```

## Troubleshooting

### MPI Not Found

```bash
# CMake
cmake -DMPI_C_COMPILER=/path/to/mpicc -DMPI_CXX_COMPILER=/path/to/mpicxx ..

# Autotools
./configure --with-mpi-dir=/path/to/mpi
```

### Build from Scratch

**CMake:**
```bash
rm -rf build
mkdir build && cd build
cmake .. && make
```

**Autotools:**
```bash
make distclean  # if Makefile exists
autoreconf -i
./configure
make
```

### Permission Denied on Install

Use custom install location:

```bash
# CMake
cmake -DCMAKE_INSTALL_PREFIX=$HOME/piafs ..

# Autotools
./configure --prefix=$HOME/piafs
```

Or use the default (installs to project directory, no special permissions needed).

## Startup Information

When PIAFS starts, it displays configuration information:

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

This helps verify:
- Execution mode (serial vs MPI)
- Number of MPI processes
- Version and commit
- Build configuration
- GPU status (if GPU-enabled build)

## Next Steps

- {doc}`running-simulations` - How to run PIAFS
- {doc}`testing` - Run the test suite
- {doc}`gpu-support` - Detailed GPU acceleration guide
