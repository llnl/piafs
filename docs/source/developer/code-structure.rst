Code Structure
==============

PIAFS is organized into modules with clear separation of concerns.

Directory Layout
----------------

Source Code
~~~~~~~~~~~

``src/``
   Main source code directory

``src/ArrayFunctions/``
   Array manipulation utilities

``src/BoundaryConditions/``
   Boundary condition implementations

``src/CommonFunctions/``
   Common utility functions

``src/FirstDerivative/``
   First derivative operators

``src/GPU/``
   GPU kernel implementations (CUDA/HIP)

``src/HyParFunctions/``
   Core solver functions

``src/InterpolationFunctions/``
   Spatial discretization schemes

``src/IOFunctions/``
   Input/output operations

``src/MathFunctions/``
   Mathematical utilities

``src/MPIFunctions/``
   MPI communication routines

``src/PhysicalModels/``
   Physics implementations (Euler, Navier-Stokes, Chemistry)

``src/TimeIntegration/``
   Time integration methods

Headers
~~~~~~~

``include/``
   Public header files

Examples and Tests
~~~~~~~~~~~~~~~~~~

``Examples/``
   Example simulations

``Tests/``
   Regression test suite

Documentation
~~~~~~~~~~~~~

``doc/``
   Additional documentation

``docs/``
   Sphinx documentation (this site)

Module Dependencies
-------------------

The code follows a layered architecture:

1. **Core** - Basic data structures and utilities
2. **Numerical** - Spatial and temporal discretization
3. **Physics** - Physical model implementations
4. **Parallel** - MPI and GPU implementations
5. **I/O** - Input/output operations

Adding New Code
---------------

When adding new features:

1. Place code in the appropriate ``src/`` subdirectory
2. Add public declarations to header files in ``include/``
3. Update CMakeLists.txt and Makefile.am
4. Add tests in ``Tests/``
5. Update documentation
