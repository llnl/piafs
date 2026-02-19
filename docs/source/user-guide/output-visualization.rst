Output and Visualization
========================

This guide covers PIAFS output formats and visualization options.

Output Formats
--------------

Text Format
~~~~~~~~~~~

ASCII text files with one row per grid point:

.. code-block:: text

   x  y  rho  rho*u  rho*v  E

Set with ``op_file_format text`` in ``solver.inp``.

**Pros:** Easy to read, compatible with all tools
**Cons:** Large file size, slower I/O

Binary Format
~~~~~~~~~~~~~

Compact binary output. Set with ``op_file_format binary``.

**Pros:** Fast I/O, small file size
**Cons:** Requires conversion for visualization

Tecplot Format
~~~~~~~~~~~~~~

2D Tecplot ASCII format. Set with ``op_file_format tecplot2d``.

**Pros:** Direct visualization in Tecplot/VisIt
**Cons:** 2D only, large files

Visualization Tools
-------------------

Python Scripts
~~~~~~~~~~~~~~

PIAFS includes Python scripts for binary format:

.. code-block:: bash

   export PIAFS_DIR=/path/to/piafs
   python $PIAFS_DIR/Examples/Python/plotSolution1D.py
   python $PIAFS_DIR/Examples/Python/plotSolution2D.py
   python $PIAFS_DIR/Examples/Python/plotSolution3D.py

Gnuplot
~~~~~~~

For text format:

.. code-block:: gnuplot

   plot 'op_00100.dat' using 1:2 with lines

MATLAB/Octave
~~~~~~~~~~~~~

.. code-block:: matlab

   data = load('op_00100.dat');
   plot(data(:,1), data(:,2))

ParaView/VisIt
~~~~~~~~~~~~~~

Use the conversion utilities in ``Extras/`` to convert binary to VTK format.

Analyzing Results
-----------------

Conservation Checks
~~~~~~~~~~~~~~~~~~~

Enable with ``conservation_check yes`` in ``solver.inp``.

PIAFS prints conservation errors for mass, momentum, and energy.

Comparing Solutions
~~~~~~~~~~~~~~~~~~~

Many examples include exact solution codes in the ``aux/`` directory.

Post-Processing
---------------

The ``Extras/`` directory contains utilities for:

* Binary to text conversion
* Binary to Tecplot conversion
* Grid file generation

Next Steps
----------

* :doc:`testing` - Verify your simulation
* :doc:`../examples/index` - Working examples with visualization
