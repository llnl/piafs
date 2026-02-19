Running Simulations
===================

This guide covers how to run PIAFS simulations in various configurations.

Basic Execution
---------------

Serial Mode
~~~~~~~~~~~

Run PIAFS without MPI:

.. code-block:: bash

   /path/to/piafs/bin/PIAFS-gcc-serial

PIAFS reads input files from the current directory by default.

Parallel Mode
~~~~~~~~~~~~~

Run with MPI using ``mpiexec``:

.. code-block:: bash

   mpiexec -n 4 /path/to/piafs/bin/PIAFS-gcc-mpi

Where ``-n 4`` specifies 4 MPI processes.

Simulation Workflow
-------------------

1. Create Working Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy an example or create a new directory with input files:

.. code-block:: bash

   cp -r Examples/1D_Euler/SodShockTube my_simulation
   cd my_simulation

2. Generate Initial Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile and run the initialization code:

.. code-block:: bash

   cd aux
   gcc init.c -lm -o INIT
   cd ..
   ./aux/INIT

This creates ``initial.inp``.

3. Run PIAFS
~~~~~~~~~~~~

.. code-block:: bash

   mpiexec -n 4 /path/to/piafs/bin/PIAFS-gcc-mpi

HPC Systems
-----------

Slurm
~~~~~

**Interactive session:**

.. code-block:: bash

   # Request resources
   salloc -N 2 -n 128 -p pbatch -t 60

   # Run simulation
   srun -n 128 /path/to/piafs/bin/PIAFS-gcc-mpi

**Batch job:**

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 2
   #SBATCH -n 128
   #SBATCH -p pbatch
   #SBATCH -t 01:00:00
   #SBATCH -J piafs_sim

   srun -n 128 /path/to/piafs/bin/PIAFS-gcc-mpi

Submit with: ``sbatch job.sh``

LSF
~~~

.. code-block:: bash

   #!/bin/bash
   #BSUB -nnodes 2
   #BSUB -W 60
   #BSUB -J piafs_sim

   jsrun -n 128 /path/to/piafs/bin/PIAFS-gcc-mpi

PBS
~~~

.. code-block:: bash

   #!/bin/bash
   #PBS -l select=2:ncpus=64:mpiprocs=64
   #PBS -l walltime=01:00:00

   mpiexec -n 128 /path/to/piafs/bin/PIAFS-gcc-mpi

GPU Execution
-------------

GPU acceleration is enabled by default for GPU builds:

.. code-block:: bash

   # GPU automatically enabled for GPU-enabled builds
   srun -n 4 --gpus-per-node=4 /path/to/piafs/bin/PIAFS-gcc-mpi

See :doc:`gpu-support` for detailed GPU configuration.

Runtime Control
---------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

* ``PIAFS_USE_GPU`` - Control GPU (1=enable [default for GPU builds], 0=disable)
* ``PIAFS_GPU_VERBOSE`` - Verbose GPU output (0/1)
* ``OMP_NUM_THREADS`` - OpenMP threads (if OpenMP enabled)

Monitoring Progress
-------------------

Screen Output
~~~~~~~~~~~~~

PIAFS prints progress to stdout:

.. code-block:: text

   Iteration:   100  Time: 0.100000  Wallclock: 2.3 sec
   Iteration:   200  Time: 0.200000  Wallclock: 4.6 sec

Frequency controlled by ``screen_op_iter`` in ``solver.inp``.

Solution Files
~~~~~~~~~~~~~~

Output files are written based on ``file_op_iter``:

* ``op_00000.dat`` - Initial solution
* ``op_00100.dat`` - Solution at iteration 100
* ``op_00200.dat`` - Solution at iteration 200

Output Formats
--------------

Text Format
~~~~~~~~~~~

ASCII text files (one row per grid point):

.. code-block:: text

   x  rho  rho*u  E

Set with ``op_file_format text`` in ``solver.inp``.

Binary Format
~~~~~~~~~~~~~

Binary output for efficiency. Set with ``op_file_format binary``.

Visualize with Python scripts in ``Examples/Python/``.

Tecplot Format
~~~~~~~~~~~~~~

2D Tecplot ASCII format. Set with ``op_file_format tecplot2d``.

Restarting Simulations
-----------------------

To continue from a previous run:

1. Set ``restart_iter`` in ``solver.inp`` to the iteration number
2. Ensure the restart file exists (e.g., ``op_01000.dat``)
3. Run PIAFS

Example:

.. code-block:: text

   begin
       restart_iter  1000
       n_iter        2000
   end

This continues from iteration 1000 to 2000.

Troubleshooting
---------------

Simulation Crashes
~~~~~~~~~~~~~~~~~~

Check:

* Input files are correct
* Initial solution exists and is valid
* Boundary conditions are properly specified
* Time step is stable (reduce ``dt`` if needed)

Performance Issues
~~~~~~~~~~~~~~~~~~

* Use appropriate number of MPI processes for problem size
* Enable OpenMP if available
* Consider GPU acceleration for large problems
* Check I/O frequency (reduce if I/O-bound)

Next Steps
----------

* :doc:`output-visualization` - Visualize results
* :doc:`gpu-support` - GPU acceleration details
* :doc:`../examples/index` - Working examples
