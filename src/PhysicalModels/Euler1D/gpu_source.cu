/*! @file gpu_source.cu
    @brief GPU kernels for Euler1D source term computation
*/

#include <gpu.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

GPU_KERNEL void gpu_euler1d_source_zero_kernel(double *source, int nvars, int npoints)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    for (int v = 0; v < nvars; v++) {
      source[idx*nvars + v] = 0.0;
    }
  }
}

