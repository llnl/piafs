2D Navier-Stokes Examples
==========================

Examples for 2D viscous compressible flow.

Available Examples
------------------

Isentropic Vortex Convection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/2D_NavierStokes/InviscidVortexConvection``

Inviscid vortex advection with exact solution. Tests:

* High-order spatial accuracy
* Vortex preservation
* Minimal numerical dissipation

Flat Plate Laminar Boundary Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/2D_NavierStokes/FlatPlateLaminar``

Laminar boundary layer over a flat plate. Compares with Blasius solution.

Lid-Driven Cavity
~~~~~~~~~~~~~~~~~

**Location:** ``Examples/2D_NavierStokes/LidDrivenCavity``

Classic viscous flow benchmark. Tests:

* No-slip wall boundary conditions
* Recirculation zones
* Steady-state convergence

2D Riemann Problem (Case 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/2D_NavierStokes/Riemann2DCase4``

2D Riemann problem with complex wave interactions.

Subsonic/Supersonic Jets
~~~~~~~~~~~~~~~~~~~~~~~~~

**Locations:**
* ``Examples/2D_NavierStokes/Uniform_Subsonic_Jet``
* ``Examples/2D_NavierStokes/Uniform_Supersonic_Jet``

Jet flows with photochemical heating.

Radial Expansion Wave
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/2D_NavierStokes/RadialExpansionWave``

Radially symmetric flow with exact solution.

Visualization
-------------

For 2D examples, use:

* Python scripts (for binary format)
* Tecplot/VisIt (for tecplot2d format)
* ParaView (after conversion)

.. code-block:: bash

   export PIAFS_DIR=/path/to/piafs
   python $PIAFS_DIR/Examples/Python/plotSolution2D.py
