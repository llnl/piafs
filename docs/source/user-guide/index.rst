User Guide
==========

This comprehensive guide covers everything you need to know about using PIAFS effectively.

Contents
--------

.. toctree::
   :maxdepth: 2

   building
   input-files
   running-simulations
   gpu-support
   boundary-conditions
   numerical-methods
   output-visualization
   testing

Overview
--------

PIAFS (Photochemically Induced Acousto-optics Fluid Simulations) is a finite-difference solver for compressible Euler and Navier-Stokes equations on Cartesian grids.

Key Capabilities
----------------

**Spatial Discretization**

* 1st through 5th order finite-difference schemes
* WENO (Weighted Essentially Non-Oscillatory) schemes
* MUSCL (Monotone Upstream-centered Scheme for Conservation Laws)
* Compact schemes

**Time Integration**

* Forward Euler
* Explicit Runge-Kutta methods (2nd, 3rd, 4th order)

**Parallelization**

* MPI for distributed-memory parallelism
* OpenMP for shared-memory parallelism
* GPU acceleration (CUDA and HIP)

**Physical Models**

* 1D Euler equations
* 2D/3D Navier-Stokes equations
* Photochemistry module for laser-induced heating

**Boundary Conditions**

* Periodic
* Extrapolation
* Dirichlet
* Slip/no-slip walls
* Subsonic inflow/outflow
* Supersonic inflow/outflow
* Sponge layers

Topics
------

:doc:`building`
   Detailed instructions for building PIAFS with CMake and Autotools, including GPU support and various build configurations.

:doc:`input-files`
   Complete reference for all input files (solver.inp, boundary.inp, physics.inp, etc.).

:doc:`running-simulations`
   How to run PIAFS in serial and parallel, on workstations and HPC systems.

:doc:`gpu-support`
   Guide to GPU acceleration with CUDA (NVIDIA) and HIP (AMD).

:doc:`boundary-conditions`
   Reference for all boundary condition types and how to use them.

:doc:`numerical-methods`
   Description of the numerical methods implemented in PIAFS.

:doc:`output-visualization`
   How to visualize and analyze PIAFS output using various tools.

:doc:`testing`
   Running the test suite and understanding test results.

Getting Help
------------

If you can't find what you're looking for:

* Check the :doc:`../examples/index` for working examples
* See the :doc:`../developer/index` if you're modifying PIAFS
* Report issues on `GitHub <https://github.com/LLNL/piafs/issues>`_
