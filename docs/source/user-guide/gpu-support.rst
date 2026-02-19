GPU Support
===========

PIAFS supports GPU acceleration through both CUDA (NVIDIA) and HIP (AMD) backends.

Building with GPU Support
--------------------------

NVIDIA CUDA
~~~~~~~~~~~

.. code-block:: bash

   cmake -DENABLE_GPU=ON -DENABLE_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=90 ..
   make -j8
   make install

AMD HIP
~~~~~~~

.. code-block:: bash

   cmake -DENABLE_GPU=ON -DENABLE_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx942 ..
   make -j8
   make install

Running with GPUs
-----------------

GPU acceleration is **enabled by default** when PIAFS is built with GPU support.

.. code-block:: bash

   # GPU is automatically enabled for GPU builds
   srun -n 4 --gpus-per-node=4 bin/PIAFS-gcc-mpi

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

* ``PIAFS_USE_GPU=0`` - Disable GPU acceleration (to run on CPU even with GPU build)
* ``PIAFS_GPU_VERBOSE=1`` - Verbose GPU output
* ``PIAFS_GPU_VALIDATE=1`` - Enable validation
* ``PIAFS_GPU_SYNC_EVERY_OP=1`` - Debug synchronization

**Note:** You only need to set ``PIAFS_USE_GPU=0`` if you want to disable GPU execution for a GPU-enabled build.

Performance Considerations
---------------------------

* Each MPI rank is assigned to a GPU based on local rank
* Data transfers between CPU and GPU occur during I/O
* Larger problems benefit more from GPU acceleration
* Consider GPU-aware MPI for multi-GPU systems

See :doc:`building` for detailed build instructions.
