API Reference
=============

This section documents the PIAFS Application Programming Interface.

.. note::
   For detailed C/C++ API documentation, you can also build the Doxygen documentation
   by running ``doxygen Doxyfile`` in the project root.

Core Components
---------------

.. toctree::
   :maxdepth: 1

   core-structures
   boundary-conditions
   time-integration
   interpolation
   physical-models
   gpu-api

Overview
--------

PIAFS is organized into several modules:

* **HyParFunctions** - Core solver routines
* **BoundaryConditions** - Boundary condition implementations
* **InterpolationFunctions** - Spatial discretization schemes
* **TimeIntegration** - Time integration methods
* **PhysicalModels** - Physics implementations (Euler, Navier-Stokes)
* **GPU** - GPU kernel implementations

Main Data Structures
--------------------

HyPar
~~~~~

Main solver structure containing:

* Grid information (dimensions, sizes)
* Solution arrays
* Boundary conditions
* Function pointers to numerical methods

Physical Model Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Euler1D`` - 1D Euler equations
* ``NavierStokes2D`` - 2D Navier-Stokes
* ``NavierStokes3D`` - 3D Navier-Stokes

Key Function Categories
-----------------------

Initialization
~~~~~~~~~~~~~~

* ``InitializeSolvers()`` - Set up solver
* ``InitializeBoundaries()`` - Configure boundaries
* ``InitialSolution()`` - Load initial condition

Time Stepping
~~~~~~~~~~~~~

* ``Solve()`` - Main time integration loop
* ``TimeForwardEuler()`` - Euler time step
* ``TimeRK()`` - Runge-Kutta time step

Spatial Discretization
~~~~~~~~~~~~~~~~~~~~~~

* ``HyperbolicFunction()`` - Compute hyperbolic terms
* ``ParabolicFunction()`` - Compute parabolic terms
* Various interpolation functions

For detailed function documentation, see the Doxygen-generated documentation
or browse the header files in ``include/``.
