PIAFS2D - Compressible Euler/Navier-Stokes Solver
------------------------------------------------------------------

`PIAFS2D` is a finite-difference code to solve the compressible Euler/Navier-Stokes
equations (with source terms) on Cartesian grids. 

Getting the code
----------------

(With SSH keys)
```
git clone ssh://git@czgitlab.llnl.gov:7999/piafs/piafs2d.git
```

(Without SSH keys)
```
https://lc.llnl.gov/gitlab/piafs/piafs2d.git
```

Documentation
-------------

Running `doxygen Doxyfile` will generate the documentation.


Compiling
---------

After downloading or cloning the code, do the following:

```
autoreconf -i
./configure
make [-j <n>] && make install
```

If these steps are successful, the binary file
```
bin/PIAFS2D
```
will be available.

Note:
+ The first command `autoreconf -i` needs to be run only the first time 
  after a fresh copy of this code is downloaded.
+ The `-j <n>` is an optional argument that will use <n> threads for compiling the code.
  For example, `make -j 4`.


Running
-------

Create a copy of an example from `Examples` directory in a new location (outside the PIAFS2D 
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

+ Run /path/to/piafs2d/bin/PIAFS2D either in serial or in parallel using `mpiexec` or `srun`.


Plotting
--------

+ If `op_file_format` is set to `binary` in `solver.inp`, the solution is written as a binary file.
  The Python scripts `Examples/Python/plotSolution*.py` can be used to generate plots. Alternatively,
  the subdirectory `Extras` has codes that can convert binary files to text and Tecplot formats.

1D:
+ If the `op_file_format` is set to `text`, the solutions are written as ASCII text files and can be
  plotted with Gnuplot or any other plotting tool.

2D:
+ If `op_file_format` is set to `tecplot2d`, the solutions are written as a 2D ASCII Tecplot file and
  can be visualized in Tecplot or VisIt.
