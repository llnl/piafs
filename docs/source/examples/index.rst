Examples
========

PIAFS includes ready-to-run examples demonstrating various capabilities.

Example Categories
------------------

.. toctree::
   :maxdepth: 1

   euler-1d
   navier-stokes-2d
   navier-stokes-3d
   chemistry

Quick Start with Examples
--------------------------

Each example directory contains:

* ``solver.inp`` - Solver parameters
* ``boundary.inp`` - Boundary conditions
* ``physics.inp`` - Physical parameters
* ``aux/`` - Initialization code

Typical Workflow
~~~~~~~~~~~~~~~~

1. Navigate to an example:

   .. code-block:: bash

      cd Examples/1D_Euler/SodShockTube

2. Generate initial solution:

   .. code-block:: bash

      cd aux
      gcc init.c -lm -o INIT
      cd ..
      ./aux/INIT

3. Run PIAFS:

   .. code-block:: bash

      mpiexec -n 4 /path/to/piafs/bin/PIAFS-gcc-mpi

4. Visualize results using tools described in :doc:`../user-guide/output-visualization`.

Examples Location
-----------------

All examples are in the ``Examples/`` directory of the PIAFS source code.

Next Steps
----------

Browse the example categories to find problems similar to your application:

* :doc:`euler-1d` - 1D inviscid flow problems
* :doc:`navier-stokes-2d` - 2D viscous flows
* :doc:`navier-stokes-3d` - 3D simulations
* :doc:`chemistry` - Photochemistry applications
