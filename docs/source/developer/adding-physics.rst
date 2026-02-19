Adding Physical Models
======================

This guide explains how to add a new physical model to PIAFS.

Overview
--------

To add a new physical model, you need to:

1. Create model structure and functions
2. Implement initialization
3. Implement flux functions
4. Add upwinding (for hyperbolic systems)
5. Add source terms (if needed)
6. Update build system

This section is under development. For now, refer to existing models:

* ``src/PhysicalModels/Euler1D/``
* ``src/PhysicalModels/NavierStokes2D/``
* ``src/PhysicalModels/NavierStokes3D/``

as templates for implementing new physics.
