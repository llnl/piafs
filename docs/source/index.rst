PIAFS Documentation
===================

**PIAFS** (Photochemically Induced Acousto-optics Fluid Simulations) is a finite-difference code to solve the compressible Euler/Navier-Stokes equations with chemical heating on Cartesian grids.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Features
--------

* Solves compressible Euler and Navier-Stokes equations in 1D, 2D, and 3D
* Multiple time integration schemes (Forward Euler, Runge-Kutta)
* High-order spatial discretization methods (WENO, MUSCL, compact schemes)
* MPI parallelization for distributed-memory systems
* GPU acceleration support (CUDA for NVIDIA, HIP for AMD)
* OpenMP support for shared-memory parallelism
* Photochemistry module for laser-induced heating
* Comprehensive test suite and examples

Quick Links
-----------

* :doc:`getting-started/installation` - Get PIAFS up and running
* :doc:`getting-started/quick-start` - Quick overview of the workflow
* :doc:`getting-started/first-simulation` - Run an example simulation
* :doc:`user-guide/index` - Comprehensive user documentation
* :doc:`examples/index` - Example problems and tutorials
* :doc:`developer/contributing` - Contribute to PIAFS development

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/index
   getting-started/installation
   getting-started/quick-start
   getting-started/first-simulation

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user-guide/index
   user-guide/building
   user-guide/input-files
   user-guide/running-simulations
   user-guide/gpu-support
   user-guide/boundary-conditions
   user-guide/numerical-methods
   user-guide/output-visualization
   user-guide/testing

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index
   examples/euler-1d
   examples/navier-stokes-2d
   examples/navier-stokes-3d
   examples/chemistry

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/core-structures
   api/boundary-conditions
   api/time-integration
   api/interpolation
   api/physical-models
   api/gpu-api

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/index
   developer/contributing
   developer/code-structure
   developer/adding-physics
   developer/adding-schemes
   developer/gpu-development

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   about/index
   about/license
   about/citation

Community and Support
---------------------

* **GitHub Repository**: https://github.com/LLNL/piafs
* **Issue Tracker**: Report bugs and request features on GitHub
* **License**: MIT License (LLNL-CODE-2015997)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
