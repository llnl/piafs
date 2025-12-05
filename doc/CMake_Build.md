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
Executable: `build/src/PIAFS`

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
| `CMAKE_INSTALL_PREFIX` | /usr/local | Installation directory |
| `MPIEXEC` | Auto-detected | MPI run command for tests (e.g., mpiexec, srun) |

## Installation

```bash
make install
```

To change install location:
```bash
cmake -DCMAKE_INSTALL_PREFIX=/your/path ..
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

For more troubleshooting help, see BUILD_CMAKE.md.
