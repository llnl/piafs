/*! @file gpu_source.cu
    @brief GPU kernels for NavierStokes3D source term computation
*/

#include <gpu.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Set source term to zero
   Each thread handles one grid point
*/
GPU_KERNEL void gpu_ns3d_source_zero_kernel(
  double *source,         /* output: source array */
  int nvars,              /* number of variables */
  int npoints             /* number of grid points */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    for (int v = 0; v < nvars; v++) {
      source[idx*nvars + v] = 0.0;
    }
  }
}

