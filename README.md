<div align="center">
<img src="piafs.jpg" alt="PIAFS Logo">

<p>
PIAFS: Photochemically Induced Acousto-optics Fluid Simulations
</p>

[Overview](#overview) -
[Code](#getting-the-code) -
[Documentation](#documentation) -
[Compiling](#compiling) -
[Running](#running) -
[Plotting](#plotting)
[Release](#release) -

</div>

## Overview

`PIAFS` is a finite-difference code to solve the compressible Euler/Navier-Stokes
equations with chemical heating on Cartesian grids. 

## Getting the Code

(With SSH keys)
```
git clone git@github.com:LLNL/piafs.git
```

(With HTTPS)
```
git clone https://github.com/LLNL/piafs.git
```

## Documentation

Running `doxygen Doxyfile` will generate the documentation.

## Compiling

PIAFS supports two build systems: **Autotools** (traditional) and **CMake** (modern).

### Option 1: CMake Build (Recommended)

CMake provides better IDE integration, faster configuration, modern tooling support, and **GPU acceleration (CUDA/HIP)**.

**Basic build with MPI:**
```
mkdir build
cd build
cmake ..
make -j 4
```

**Serial build (no MPI):**
```
mkdir build
cd build
cmake -DENABLE_SERIAL=ON ..
make -j 4
```

**Build with OpenMP support:**
```
mkdir build
cd build
cmake -DENABLE_OMP=ON ..
make -j 4
```

**Build with CUDA GPU support (NVIDIA):**
```
mkdir build
cd build
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<CUDA architecture> ..
make -j 4
```

**Build with HIP GPU support (AMD):**
```
mkdir build
cd build
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<AMD architecture> ..
make -j 4
```

**Quick build for Matrix (NVIDIA H100):**
```
cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 .. && make -j8
```

**Quick build for Tuolumne (AMD MI300A):**
```
cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 .. && make -j8
```

If successful, the executable will be at `build/src/PIAFS-<compiler>-<mpi/serial>` (e.g., `PIAFS-gcc-mpi`).

The binary name includes:
- Compiler name (e.g., `gcc`, `clang`, `intel`)
- MPI mode: `mpi` for parallel builds or `serial` for serial builds
- OpenMP suffix if enabled (e.g., `-omp`)
- Build type suffix for non-Release builds (e.g., `-debug`)

For detailed CMake build instructions, GPU support, and machine-specific examples, see `doc/CMake_Build.md`.

### Option 2: Autotools Build

**Note:** Autotools builds do **not** support GPU acceleration (CUDA/HIP). For GPU support, use the CMake build system.

After downloading or cloning the code, do the following:

```
autoreconf -i
./configure
make [-j <n>] && make install
```

If these steps are successful, the binary file
```
bin/PIAFS-<compiler>-<mpi/serial>
```
will be available (e.g., `bin/PIAFS-gcc-mpi`).

The binary name includes:
- Compiler name (e.g., `gcc`, `clang`, `intel`)
- MPI mode: `mpi` for parallel builds or `serial` for serial builds
- OpenMP suffix if enabled (e.g., `-omp`)

Note:
+ The first command `autoreconf -i` needs to be run the first time
  after a fresh copy of this code is downloaded and any other time
  when there are major changes (addition/deletion of new source files
  and/or subdirectories).
+ The `-j <n>` is an optional argument that will use <n> threads for compiling the code.
  For example, `make -j 4`.

### Build System Comparison

| Feature | Autotools | CMake |
|---------|-----------|-------|
| Configuration | `./configure` | `cmake ..` |
| Serial mode | `--enable-serial` | `-DENABLE_SERIAL=ON` |
| OpenMP | `--enable-omp` | `-DENABLE_OMP=ON` |
| MPI executor | `--with-mpiexec=srun` | `-DMPIEXEC=srun` |
| Testing | `make check` | `make test` or `ctest` |
| IDE Integration | Limited | Excellent |
| Out-of-tree builds | Supported | Native |

## Testing

PIAFS includes a regression test suite that validates simulation outputs against benchmark solutions.

### Running Tests

**With CMake:**
```bash
cd build
make test
# or
ctest --verbose
```

**With Autotools:**
```bash
make check
```

### HPC Platform Configuration

For systems using job schedulers (Slurm, etc.), specify the MPI launch command:

**CMake:**
```bash
cmake -DMPIEXEC="srun" ..
make test
```

**Autotools:**
```bash
./configure --with-mpiexec="srun"
make check
```

For detailed testing documentation, see `Tests/README.md`.

## Running

Create a copy of an example from `Examples` directory in a new location (outside the PIAFS 
directory), then follow these steps:

+ Compile the code `<example_dir>/aux/init.[c,C]` as follows:
```
gcc init.c -lm -o INIT
```
or
```
g++ init.C -o INIT
```
depending on the whether the file is `.c` or `.C`.

+ In the run directory (that contains the `solver.inp`, `boundary.inp`, etc), run
  the binary `INIT` from the previous step. This will generate the initial solution.

+ Run `/path/to/piafs/bin/PIAFS-<compiler>-<mpi/serial>` either in serial or in parallel using `mpiexec` or `srun`.

  **Note:** With CMake, the default install location is the source directory (matching autotools), so after `make install`, the binary will be at `piafs/bin/PIAFS-<compiler>-<mpi/serial>` (e.g., `PIAFS-gcc-mpi`).

  **Startup Information:** When PIAFS starts, it prints detailed build information including:
  - Version and Git hash/branch
  - Build date
  - Compiler name and version
  - MPI mode and version (for parallel builds)
  - OpenMP status
  - Build type


## Plotting

+ If `op_file_format` is set to `binary` in `solver.inp`, the solution is written as a binary file.
  The Python scripts `Examples/Python/plotSolution*.py` can be used to generate plots. 
  Alternatively, the subdirectory `Extras` has codes that can convert binary files to text and 
  Tecplot formats.

Note that the Python scripts expect the following environment to be defined:
```
PIAFS_DIR=/path/to/piafs
```

1D:
+ If the `op_file_format` is set to `text`, the solutions are written as ASCII text files and can be
  plotted with Gnuplot or any other plotting tool.

2D:
+ If `op_file_format` is set to `tecplot2d`, the solutions are written as a 2D ASCII Tecplot file and
  can be visualized in Tecplot or VisIt.

## Funding

This work was funded by the Laboratory Research and Development Program at LLNL under Project Tracking Code No. 24-ERD-001.

## Release

LLNL-CODE-2015997
