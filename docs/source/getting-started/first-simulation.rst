Example Simulation
==================

This tutorial walks you through running a PIAFS simulation that demonstrates one of its unique capabilities: modeling photochemically induced acoustic waves.

Beam-Induced Density Modulation
--------------------------------

This example simulates density modulation induced by photo-chemical heating from a laser beam. It demonstrates the coupling between fluid dynamics and chemistry that makes PIAFS particularly useful for gaseous optics applications.

**Physical Process:**

1. UV laser photons are absorbed by ozone (O₃) molecules
2. Photodissociation and chemical reactions release heat
3. Heating creates pressure and density perturbations
4. Acoustic waves propagate through the gas

This showcases PIAFS's unique feature: integrated photochemistry and hydrodynamics for gaseous optics research.

Why This Example?
~~~~~~~~~~~~~~~~~

* Demonstrates PIAFS's core capability (photochemistry)
* Relevant to real gaseous optics experiments
* Shows chemistry-fluid coupling
* All input files provided

Step-by-Step Tutorial
---------------------

Step 1: Navigate to the Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /path/to/piafs/Examples/1D_Euler/BeamInducedDensityMod

Step 2: Examine the Input Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example requires four input files plus the initial solution.

**solver.inp** - Main solver parameters:

.. code-block:: text

   begin
       ndims               1
       nvars               24
       size                256
       n_iter              785
       time_scheme         rk
       hyp_space_scheme    weno5
       dt                  0.04
       screen_op_iter      5
       file_op_iter        5
       op_file_format      binary
       model               euler1d
   end

Key parameters:

* ``ndims = 1``: 1D problem
* ``nvars = 24``: 24 variables (includes chemical species)
* ``hyp_space_scheme = weno5``: 5th order WENO for shock capturing
* ``model = euler1d``: 1D Euler equations with chemistry

**boundary.inp** - Boundary conditions:

.. code-block:: text

   2
   extrapolate  0  1
   extrapolate  0 -1

Both boundaries use extrapolation (zero-gradient outflow).

**physics.inp** - Physical parameters:

.. code-block:: text

   begin
       gamma             1.4
       include_chemistry yes
   end

* ``gamma = 1.4``: Ratio of specific heats for air
* ``include_chemistry = yes``: Enable photochemistry module

**chemistry.inp** - Photochemistry parameters:

.. code-block:: text

   begin
       lambda_UV   2.48e-7
       theta       0.0029671
       f_O3        0.01
       Ptot        101325.0
       Ti          288
       t_pulse     1e-8
       F0          2000.0
   end

Key chemistry parameters:

* ``lambda_UV``: UV wavelength (248 nm)
* ``f_O3``: Ozone mole fraction (1%)
* ``Ptot``: Total pressure (101325 Pa, 1 atm)
* ``Ti``: Initial temperature (288 K)
* ``t_pulse``: Laser pulse duration (10 ns)
* ``F0``: Laser fluence (2000 J/m²)

Step 3: Generate Initial Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial solution code sets up the uniform gas state:

.. code-block:: bash

   cd aux
   gcc init.c -lm -o INIT
   cd ..
   ./aux/INIT

This creates ``initial.inp`` with:

* Uniform density, pressure, and temperature
* Zero initial velocity
* Equilibrium chemical species concentrations

Step 4: Run PIAFS
~~~~~~~~~~~~~~~~~

**Serial execution:**

.. code-block:: bash

   /path/to/piafs/bin/PIAFS-gcc-serial

**Parallel execution (4 processes):**

.. code-block:: bash

   mpiexec -n 4 /path/to/piafs/bin/PIAFS-gcc-mpi

You should see output like:

.. code-block:: text

   ================================================================================
   PIAFS - Parallel (MPI) version with 4 processes
     Version: 0.1
     Build Type: Release
   ================================================================================

   Reading solver inputs from file "solver.inp".
   Reading boundary conditions from file "boundary.inp".
   Reading physics inputs from file "physics.inp".
   Reading chemistry inputs from file "chemistry.inp".

   Iteration:    5  Time: 0.200  Wallclock: 1.2 sec
   Iteration:   10  Time: 0.400  Wallclock: 2.4 sec
   ...
   Iteration:  785  Time: 31.400  Wallclock: 45.3 sec

   Completed simulation.

Step 5: Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

Since we set ``op_file_format = binary``, use the Python plotting scripts:

.. code-block:: bash

   export PIAFS_DIR=/path/to/piafs
   python $PIAFS_DIR/Examples/Python/plotSolution1D.py

**Alternative: Convert to text format**

You can change ``op_file_format`` to ``text`` in ``solver.inp`` and rerun, then plot with:

.. code-block:: bash

   gnuplot

At the gnuplot prompt:

.. code-block:: gnuplot

   plot 'op_00785.dat' using 1:2 with lines title 'Density'

Understanding the Results
--------------------------

The simulation shows:

**Density Modulation**
   The laser heating creates a periodic density pattern that evolves into acoustic waves propagating away from the heated regions.

**Acoustic Waves**
   Pressure perturbations launch sound waves that travel at the speed of sound in the gas.

**Chemical Heating**
   The photochemistry module tracks ozone photodissociation and subsequent exothermic reactions that heat the gas.

**Entropy Mode**
   A non-propagating temperature/entropy modulation remains at the original heating location.

Physical Significance
~~~~~~~~~~~~~~~~~~~~~

This simulation is directly relevant to:

* **Gaseous optics** - Creating refractive index gratings in gas
* **Photoacoustic imaging** - Laser-induced acoustic wave generation
* **Inertial fusion energy** - Gaseous optical elements for IFE facilities
* **Atmospheric chemistry** - Ozone photochemistry in the upper atmosphere

Modifying the Simulation
-------------------------

Try experimenting with different parameters:

**Increase ozone concentration:**

In ``chemistry.inp``, change ``f_O3`` to 0.02 (2%):

.. code-block:: text

   f_O3  0.02

This increases the heating and produces stronger acoustic waves.

**Change laser fluence:**

Modify ``F0`` in ``chemistry.inp`` to change the laser energy:

.. code-block:: text

   F0  4000.0

Higher fluence creates stronger perturbations.

**Use different numerical scheme:**

In ``solver.inp``, try other schemes:

.. code-block:: text

   hyp_space_scheme  muscl3    # 3rd order MUSCL

**Change resolution:**

Increase grid resolution for better accuracy:

.. code-block:: text

   size  512

Remember to regenerate ``initial.inp`` after changing parameters!

Next Steps
----------

Now that you've run a chemistry example, explore more:

* :doc:`../examples/euler-1d` - More 1D examples
* :doc:`../examples/chemistry` - Advanced chemistry cases
* :doc:`../examples/navier-stokes-2d` - 2D viscous flows with chemistry
* :doc:`../user-guide/input-files` - Detailed input file reference
* :doc:`../user-guide/numerical-methods` - Understanding the methods
