Adding Numerical Schemes
========================

This guide explains how to add new spatial or temporal discretization schemes.

Adding a Spatial Scheme
------------------------

1. Create implementation file in ``src/InterpolationFunctions/``
2. Add function declaration to ``include/interpolation.h``
3. Register scheme in ``InitializeSolvers()``
4. Add tests

Refer to existing schemes as templates:

* ``src/InterpolationFunctions/Interp1PrimFifthOrderWENO.c``
* ``src/InterpolationFunctions/Interp1PrimSecondOrderMUSCL.c``

Adding a Time Integration Scheme
---------------------------------

1. Create implementation in ``src/TimeIntegration/``
2. Add declaration to ``include/timeintegration.h``
3. Register in time integration initialization
4. Add tests

This section is under development. Refer to existing implementations for guidance.
