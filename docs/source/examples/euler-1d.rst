1D Euler Examples
=================

Examples for 1D inviscid compressible flow.

Available Examples
------------------

Sod Shock Tube
~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/SodShockTube``

Classic shock tube problem with exact solution.

* Shock wave propagation
* Contact discontinuity
* Expansion fan
* Tests shock-capturing capability

Lax Shock Tube
~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/LaxShockTube``

Alternative shock tube configuration.

Shu-Osher Problem
~~~~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/ShuOsherProblem``

Shock-entropy wave interaction testing high-order schemes.

Density Sine Wave
~~~~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/DensitySineWave``

Smooth density wave advection with exact solution.

Pressure Perturbation
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/PressurePerturbation``

Acoustic wave propagation.

Beam-Induced Density Modulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/BeamInducedDensityMod``

Photochemistry example with laser-induced heating.

Requires ``chemistry.inp``.

Running an Example
------------------

Using the Sod shock tube as an example:

.. code-block:: bash

   cd Examples/1D_Euler/SodShockTube
   cd aux && gcc init.c -lm -o INIT && cd ..
   ./aux/INIT
   mpiexec -n 4 /path/to/piafs/bin/PIAFS-gcc-mpi

Visualize with:

.. code-block:: bash

   gnuplot
   > plot 'op_00200.dat' using 1:2 with lines

See :doc:`../getting-started/first-simulation` for a detailed walkthrough.
