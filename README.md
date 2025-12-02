<div align="center">
<img src="piafs.jpg" alt="PIAFS Logo">

<p>
PIAFS: Photochemically Induced Acousto-optics Fluid Simulations
</p>

[Overview](#Overview) -
[Code](#Getting the Code) -
[Documentation](#Documentation) -
[Compiling](#Compiling) -
[Running](#Running) -
[Plotting](#Plotting) -

</div>

## Overview

`PIAFS` is a finite-difference code to solve the compressible Euler/Navier-Stokes
equations with chemical heating on Cartesian grids. 

## Getting the Code

(With SSH keys)
```
git clone ssh://git@czgitlab.llnl.gov:7999/piafs/piafs.git
```

(Without SSH keys)
```
https://lc.llnl.gov/gitlab/piafs/piafs.git
```

## Documentation

Running `doxygen Doxyfile` will generate the documentation.

## Compiling

PIAFS supports two build systems: **Autotools** (traditional) and **CMake** (modern).

### Option 1: CMake Build (Recommended)

CMake provides better IDE integration, faster configuration, and modern tooling support.

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

If successful, the executable will be at `build/src/PIAFS`.

For detailed CMake build instructions and options, see [BUILD_CMAKE.md](BUILD_CMAKE.md).

### Option 2: Autotools Build

After downloading or cloning the code, do the following:

```
autoreconf -i
./configure
make [-j <n>] && make install
```

If these steps are successful, the binary file
```
bin/PIAFS
```
will be available.

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
| IDE Integration | Limited | Excellent |
| Out-of-tree builds | Supported | Native |

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

+ Run /path/to/piafs/bin/PIAFS either in serial or in parallel using `mpiexec` or `srun`.


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
