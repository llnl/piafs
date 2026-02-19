3D Navier-Stokes Examples
==========================

Examples for 3D viscous compressible flow.

Available Examples
------------------

3D Vortex Convection
~~~~~~~~~~~~~~~~~~~~

**Locations:**
* ``Examples/3D_NavierStokes/VortexConvection_XY``
* ``Examples/3D_NavierStokes/VortexConvection_XZ``
* ``Examples/3D_NavierStokes/VortexConvection_YZ``

Isentropic vortex in different planes.

Density Sine Wave
~~~~~~~~~~~~~~~~~

**Location:** ``Examples/3D_NavierStokes/DensitySineWave``

3D density wave advection.

Direct Numerical Simulation of Turbulence Decay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/3D_NavierStokes/DNS_IsotropicTurbulenceDecay``

Isotropic turbulence decay simulation. Requires:

* Large grid (128³ or higher)
* Parallel execution
* Significant computational resources

2D Riemann in 3D
~~~~~~~~~~~~~~~~

**Locations:**
* ``Examples/3D_NavierStokes/2D_RiemannCase4_XY``
* ``Examples/3D_NavierStokes/2D_RiemannCase4_XZ``
* ``Examples/3D_NavierStokes/2D_RiemannCase4_YZ``

2D Riemann problems in different planes (tests 3D implementation).

Running 3D Examples
-------------------

3D examples typically require more resources:

.. code-block:: bash

   cd Examples/3D_NavierStokes/VortexConvection_XY
   cd aux && gcc init.c -lm -o INIT && cd ..
   ./aux/INIT
   mpiexec -n 16 /path/to/piafs/bin/PIAFS-gcc-mpi

Visualization
-------------

.. code-block:: bash

   export PIAFS_DIR=/path/to/piafs
   python $PIAFS_DIR/Examples/Python/plotSolution3D.py
