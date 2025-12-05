# Building PIAFS with Autotools {#autotools_build}

[TOC]

## Overview

PIAFS uses the GNU Autotools build system (autoconf/automake) as its traditional build method. Autotools provides:
- Portable configuration across Unix-like systems
- Standard GNU build conventions
- Automatic dependency detection
- Well-established build patterns

## Quick Start

### Default Build (with MPI)

```bash
autoreconf -i
./configure
make -j4
make install
```

The executable will be installed at: `bin/PIAFS`

**Note:** The first command `autoreconf -i` is only needed:
- After a fresh clone/download of the code
- When source files or subdirectories are added/removed
- After major structural changes

### Common Build Variants

```bash
# Serial (no MPI)
./configure --enable-serial

# With OpenMP
./configure --enable-omp

# With custom MPI directory
./configure --with-mpi-dir=/path/to/mpi

# Custom installation prefix
./configure --prefix=/path/to/install

# Debug build
CFLAGS="-g -O0" CXXFLAGS="-g -O0" ./configure
```

## Configure Options Summary

### PIAFS-Specific Options

| Option | Description |
|--------|-------------|
| `--enable-serial` | Build without MPI (serial mode) |
| `--enable-omp` | Enable OpenMP support |
| `--with-mpi-dir=DIR` | Specify MPI installation directory |
| `--with-mpiexec=CMD` | Specify MPI run command for tests (e.g., srun, jsrun) |

### Standard Autotools Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prefix=PREFIX` | `$(pwd)` | Installation directory |
| `--help` | - | Show all configuration options |

### Environment Variables

Set these before running `./configure`:

```bash
# Compiler flags
CFLAGS="-O3 -march=native"
CXXFLAGS="-O3 -march=native"

# Compilers (if MPI not in PATH)
CC=/path/to/mpicc
CXX=/path/to/mpicxx
```

## Installation

By default, autotools installs to the project's own directory:

```bash
make install
```

This creates:
- `bin/PIAFS` - Executable

To install to a different location:

```bash
./configure --prefix=/usr/local
make install
```

## Testing

PIAFS includes a comprehensive regression test suite that compares simulation outputs against benchmark solutions.

### Running Tests

```bash
# Run all tests
make check

# View detailed test output
cat Tests/test_mpi.sh.log
```

### Custom MPI Executor (HPC Platforms)

For HPC systems with job schedulers, specify a custom MPI launch command during configuration:

```bash
# For Slurm
./configure --with-mpiexec="srun"

# For IBM JSRun
./configure --with-mpiexec="jsrun"

# For standard mpirun with options
./configure --with-mpiexec="mpirun -np"

# Then run tests
make check
```

### Out-of-Source Testing

The test system supports out-of-source builds:

```bash
mkdir build && cd build
../configure --with-mpiexec="srun"
make
make check
```

### Test Output

Test results include:
- Summary of passed/failed file comparisons
- Detailed diff output for any failures
- Test logs saved to `Tests/test_mpi.sh.log`
- Benchmark comparisons use relative tolerance of 1.0e-14

For more details, see `Tests/README.md` in the source directory.

## Common Workflows

### Development Build

```bash
autoreconf -i
CFLAGS="-g -O0" CXXFLAGS="-g -O0" ./configure
make -j4
make install
```

### Production Build

```bash
autoreconf -i
CFLAGS="-O3 -march=native" CXXFLAGS="-O3 -march=native" ./configure
make -j4
make install
```

### Serial Build (No MPI)

```bash
autoreconf -i
./configure --enable-serial
make -j4
make install
```

### Parallel Build with OpenMP

```bash
autoreconf -i
./configure --enable-omp --with-mpi-dir=/opt/openmpi
make -j4
make install
```

## Cleaning Up

```bash
# Clean build artifacts
make clean

# Remove all generated files (requires reconfigure)
make distclean

# Complete cleanup (requires autoreconf -i again)
make maintainer-clean
```

## Troubleshooting

### MPI Not Found

**Problem:** Configure cannot find MPI compilers

**Solution:**
```bash
# Specify MPI directory
./configure --with-mpi-dir=/path/to/mpi

# Or set environment variables
export MPI_DIR=/path/to/mpi
./configure

# Or build in serial mode
./configure --enable-serial
```

### Configure Script Not Found

**Problem:** `./configure: No such file or directory`

**Solution:**
```bash
# Generate configure script
autoreconf -i
./configure
```

### Permission Denied on Install

**Problem:** `make install` fails with permission errors

**Solution:**
```bash
# Install to custom location (no sudo needed)
./configure --prefix=$HOME/piafs
make install

# Or use default (installs to project directory)
./configure
make install
```

### Build from Scratch

To start completely fresh:
```bash
make distclean  # if Makefile exists
autoreconf -i
./configure
make -j4
make install
```

## Comparison with CMake

| Feature | Autotools | CMake |
|---------|-----------|-------|
| Configure | `./configure` | `cmake ..` |
| Out-of-tree builds | Supported | Native |
| Parallel builds | `make -jN` | `make -jN` |
| Testing | `make check` | `make test` or `ctest` |
| MPI executor | `--with-mpiexec=srun` | `-DMPIEXEC=srun` |
| Clean build | `make distclean` | `rm -rf build` |
| IDE integration | Limited | Excellent |
| Cross-platform | Unix-like | All platforms |

For CMake build instructions, see \ref cmake_build "Building PIAFS with CMake".

## Advanced Configuration

### Viewing All Options

```bash
./configure --help
```

### Configuration Status

The configure script displays:
- Detected compilers (C, C++)
- MPI status
- Build mode (serial/parallel)
- OpenMP status

Review this output to verify your configuration.

### Generated Files

After configuration, key files are:
- `Makefile` - Build instructions
- `config.h` - Configuration header
- `config.log` - Detailed configuration log (useful for debugging)

## Build System Files

The autotools build system uses:
- `configure.ac` - Autoconf configuration
- `Makefile.am` - Automake makefiles (in each directory)
- `autoreconf` - Generates configure script

When adding source files, update the appropriate `Makefile.am`.

## Additional Resources

For general autotools documentation:
- [GNU Autoconf Manual](https://www.gnu.org/software/autoconf/manual/)
- [GNU Automake Manual](https://www.gnu.org/software/automake/manual/)

For PIAFS-specific information:
- See README.md for project overview
- See \ref cmake_build for CMake build alternative
