GPU Development
===============

Guide for developing GPU kernels in PIAFS.

GPU Architecture
----------------

PIAFS supports:

* **CUDA** for NVIDIA GPUs
* **HIP** for AMD GPUs

Code is organized in ``src/GPU/`` with:

* CUDA implementations (``.cu`` files)
* HIP implementations
* Common interfaces

Writing GPU Kernels
-------------------

1. Create kernel in ``src/GPU/``
2. Add declaration to appropriate ``include/gpu*.h``
3. Implement both CUDA and HIP versions if possible
4. Add memory management
5. Test on target hardware

Best Practices
--------------

* Minimize host-device data transfers
* Use appropriate block/grid sizes
* Profile performance
* Validate results against CPU version

This section is under development. Refer to existing GPU kernels in ``src/GPU/`` for examples.
