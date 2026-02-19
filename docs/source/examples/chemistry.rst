Chemistry Examples
==================

Examples with photochemistry and laser-induced heating.

Photochemistry Module
---------------------

PIAFS includes a photochemistry module for simulating laser-induced acoustic waves in gas mixtures.

Physical Model
~~~~~~~~~~~~~~

* UV laser pulse absorption by ozone (O₃)
* Chemical reactions releasing heat
* Acoustic wave generation
* Gas mixture (O₃, O₂, O, CO₂, N₂)

Configuration
~~~~~~~~~~~~~

Requires ``chemistry.inp`` with parameters:

* ``lambda_UV`` - UV wavelength
* ``F0`` - Laser fluence
* ``f_O3`` - Ozone mole fraction
* ``Ptot`` - Total pressure
* Chemical reaction rates

Available Examples
------------------

1D Beam-Induced Density Modulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``Examples/1D_Euler/BeamInducedDensityMod``

1D acoustic wave generation from laser heating.

2D Beam-Induced Density Modulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Locations:**
* ``Examples/2D_NavierStokes/BeamInducedDensityMod_1D_x``
* ``Examples/2D_NavierStokes/Inviscid_BeamInducedDensityMod_1D_x``

2D simulations of laser-induced acoustics (viscous and inviscid).

Subsonic/Supersonic Jets with Heating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Locations:**
* ``Examples/2D_NavierStokes/Uniform_Subsonic_Jet``
* ``Examples/2D_NavierStokes/Uniform_Supersonic_Jet``

Jet flows with photochemical heat release.

3D Examples
~~~~~~~~~~~

**Locations:**
* ``Examples/3D_NavierStokes/BeamInducedDensityMod_1D_x``
* ``Examples/3D_NavierStokes/DensityFluctuations``

3D simulations with chemistry.

Running Chemistry Examples
--------------------------

.. code-block:: bash

   cd Examples/1D_Euler/BeamInducedDensityMod
   cd aux && gcc init.c -lm -o INIT && cd ..
   ./aux/INIT
   mpiexec -n 4 /path/to/piafs/bin/PIAFS-gcc-mpi

The ``chemistry.inp`` file is already provided in each example.

Input Files
-----------

In addition to standard input files, chemistry examples include:

* ``chemistry.inp`` - Photochemistry parameters
* ``physics.inp`` with ``include_chemistry yes``

See :doc:`../user-guide/input-files` for parameter descriptions.

Applications
------------

This module is useful for:

* Laser-induced acoustic imaging
* Photoacoustic spectroscopy
* Optoacoustic phenomena
* Chemical kinetics in flows

References
----------

For the physics and chemistry models, see the PIAFS publications and documentation.
