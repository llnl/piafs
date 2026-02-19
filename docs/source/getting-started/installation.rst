Installation
============

This guide will help you install and build PIAFS on your system.

Getting the Code
----------------

Clone the PIAFS repository from GitHub:

**With SSH keys:**

.. code-block:: bash

   git clone git@github.com:LLNL/piafs.git
   cd piafs

**With HTTPS:**

.. code-block:: bash

   git clone https://github.com/LLNL/piafs.git
   cd piafs

Prerequisites
-------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **C/C++ Compiler**: GCC, Clang, or Intel compiler
* **MPI Library** (for parallel builds): OpenMPI, MPICH, or Intel MPI

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **CUDA Toolkit** (for NVIDIA GPU support)
* **ROCm/HIP** (for AMD GPU support)
* **OpenMP** (for shared-memory parallelism)

Building PIAFS
--------------

PIAFS supports two build systems: **CMake** (recommended) and **Autotools** (traditional).

CMake Build (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

CMake provides better IDE integration, GPU support, and modern tooling.

**Basic build with MPI:**

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make -j 4
   make install

The executable will be at ``bin/PIAFS-<compiler>-mpi`` (e.g., ``PIAFS-gcc-mpi``).

**Serial build (no MPI):**

.. code-block:: bash

   mkdir build
   cd build
   cmake -DENABLE_SERIAL=ON ..
   make -j 4
   make install

**With OpenMP:**

.. code-block:: bash

   cmake -DENABLE_OMP=ON ..
   make -j 4
   make install

**With CUDA GPU support (NVIDIA):**

.. code-block:: bash

   cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..
   make -j 4
   make install

Replace ``90`` with your GPU's compute capability (80 for A100, 75 for RTX 20xx, etc.).

**With HIP GPU support (AMD):**

.. code-block:: bash

   cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 ..
   make -j 4
   make install

Replace ``gfx942`` with your GPU architecture (gfx90a for MI250X, gfx908 for MI100, etc.).

For more detailed CMake instructions, see :doc:`../user-guide/building`.

Autotools Build
~~~~~~~~~~~~~~~

**Note:** Autotools does not support GPU acceleration. Use CMake for GPU builds.

.. code-block:: bash

   autoreconf -i
   ./configure
   make -j 4
   make install

The executable will be at ``bin/PIAFS-<compiler>-mpi``.

For serial builds:

.. code-block:: bash

   ./configure --enable-serial
   make -j 4
   make install

LLNL LC Machine-Specific Instructions
--------------------------------------

If you're building PIAFS on LLNL Livermore Computing (LC) systems, use these machine-specific instructions.

Matrix (NVIDIA H100 GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Matrix has NVIDIA H100 GPUs with compute capability 9.0.

**Build:**

.. code-block:: bash

   cd ~/piafs
   rm -rf build
   mkdir build
   cd build

   cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 ..

   make -j8
   make install

**Run:**

.. code-block:: bash

   # Request interactive session
   salloc -N 1 --gpus-per-node=4 -p pbatch -t 60

   # Run with 4 MPI ranks, 4 GPUs (GPU enabled by default for GPU builds)
   srun -n 4 --gpus-per-node=4 ~/piafs/bin/PIAFS-gcc-mpi

Tuolumne (AMD MI300A GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tuolumne has AMD MI300A GPUs with architecture gfx942.

**Build:**

.. code-block:: bash

   cd ~/piafs
   rm -rf build
   mkdir build
   cd build

   # MPI wrapper is already configured with rocmcc
   cmake -DENABLE_GPU=ON -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 ..

   make -j8
   make install

**Run:**

.. code-block:: bash

   # Request interactive session
   salloc -N 1 --gpus-per-node=4 -p mi300a -t 60

   # Run with 4 MPI ranks, 4 GPUs (GPU enabled by default for GPU builds)
   srun -n 4 --gpus-per-node=4 ~/piafs/bin/PIAFS-gcc-mpi

Dane (CPU Only)
~~~~~~~~~~~~~~~

Dane is a CPU-only cluster with Intel Xeon processors.

**Build:**

.. code-block:: bash

   cd ~/piafs
   rm -rf build
   mkdir build
   cd build

   cmake -DCMAKE_BUILD_TYPE=Release ..

   make -j8
   make install

**Run:**

.. code-block:: bash

   # Request interactive session
   salloc -N 2 -p pbatch -t 60

   # Run with 128 MPI ranks (64 per node)
   srun -N 2 -n 128 ~/piafs/bin/PIAFS-gcc-mpi

**With OpenMP (hybrid MPI+OpenMP):**

.. code-block:: bash

   # Build with OpenMP
   cmake -DENABLE_OMP=ON -DCMAKE_BUILD_TYPE=Release ..
   make -j8
   make install

   # Run: 4 MPI ranks per node, 16 OpenMP threads per rank
   export OMP_NUM_THREADS=16
   srun -N 2 -n 8 -c 16 ~/piafs/bin/PIAFS-gcc-mpi

Common GPU Architectures Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NVIDIA (CUDA):**

* ``90`` - H100 (Hopper) - Matrix
* ``80`` - A100 (Ampere)
* ``75`` - RTX 20xx/Titan RTX (Turing)
* ``70`` - V100 (Volta)

Find your GPU's compute capability:

.. code-block:: bash

   nvidia-smi --query-gpu=compute_cap --format=csv,noheader

**AMD (HIP):**

* ``gfx942`` - MI300A/MI300X - Tuolumne
* ``gfx90a`` - MI250X/MI250
* ``gfx908`` - MI100
* ``gfx906`` - MI50/MI60

Find your GPU architecture:

.. code-block:: bash

   rocminfo | grep "Name:" | grep gfx

Verifying the Installation
---------------------------

After building, verify the installation by running:

.. code-block:: bash

   ./bin/PIAFS-gcc-mpi

You should see startup information including version, build date, and configuration.

Running Tests
~~~~~~~~~~~~~

PIAFS includes a comprehensive test suite:

**With CMake:**

.. code-block:: bash

   cd build
   make test

**With Autotools:**

.. code-block:: bash

   make check

All tests should pass. If any tests fail, check your build configuration.

Next Steps
----------

* :doc:`quick-start` - Learn the basic PIAFS workflow
* :doc:`first-simulation` - Run your first simulation
* :doc:`../user-guide/building` - Detailed build options and configurations
