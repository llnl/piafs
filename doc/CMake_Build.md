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
Executable: `build/src/PIAFS-v<VERSION>-<compiler><version>-<mpi><version>` (e.g., `PIAFS-v0.1-gcc9-mpi4`)

The binary name automatically includes version, compiler, and MPI information to distinguish different builds.

### Common Build Variants
```bash
# Serial (no MPI)
cmake -DENABLE_SERIAL=ON ..

# With OpenMP
cmake -DENABLE_OMP=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Build Options Summary

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_SERIAL` | OFF | Build without MPI |
| `ENABLE_OMP` | OFF | Enable OpenMP |
| `CMAKE_BUILD_TYPE` | Release | Debug, Release, RelWithDebInfo, MinSizeRel |
| `CMAKE_INSTALL_PREFIX` | Source directory | Installation directory (matches autotools) |
| `MPIEXEC` | Auto-detected | MPI run command for tests (e.g., mpiexec, srun) |

## Installation

By default, CMake installs to the project's own directory (matching autotools behavior):

```bash
make install
```

This creates:
- `bin/PIAFS-v<VERSION>-<compiler><version>-<mpi><version>` - Executable

The binary name format is: `PIAFS-v<VERSION>-<compiler><version>-<mpi><version>[-omp][-<buildtype>]`

Examples:
- `PIAFS-v0.1-gcc9-mpi4` - Version 0.1, GCC 9, MPI 4, Release build
- `PIAFS-v0.1-clang19-mpi4-omp` - Version 0.1, Clang 19, MPI 4, OpenMP enabled
- `PIAFS-v0.1-gcc9-mpi4-debug` - Version 0.1, GCC 9, MPI 4, Debug build

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

When PIAFS starts, it displays detailed build information:

```
================================================================================
PIAFS - Parallel (MPI) version with 64 processes
  Version: 0.1
  Git Hash: abc1234 (branch: main)
  Build Date: 2024-01-15 10:30:45
  Compiler: gcc 9.4.0
  MPI: parallel 4.0.0
  Build Type: Release
  OpenMP: enabled
================================================================================
```

This information helps identify:
- Which version of the code is running
- Build configuration (compiler, MPI, OpenMP)
- Git commit information for reproducibility

For more troubleshooting help, see BUILD_CMAKE.md.
