/*! @file gpu_flux.cu
    @brief GPU kernels and launch wrappers for Euler1D flux computation
*/

#include <gpu.h>
#include <gpu_flux.h>
#include <physicalmodels/euler1d.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

#define DEFAULT_BLOCK_SIZE 256

/* Kernel: Compute flux for Euler1D */
GPU_KERNEL void gpu_euler1d_flux_kernel(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma
)
{
  int total_points = 1;
  for (int i = 0; i < ndims; i++) {
    total_points *= (dim[i] + 2 * ghosts);
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_points) {
    /* idx is already the flat index into the array with ghosts, so p = idx */
    int p = idx;
    
    double rho = u[p*nvars + 0];
    double v = (rho == 0) ? 0 : u[p*nvars + 1] / rho;
    double e = u[p*nvars + 2];
    double vsq = v*v;
    double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
    
    f[p*nvars + 0] = rho * v;
    f[p*nvars + 1] = rho * v * v + P;
    f[p*nvars + 2] = (e + P) * v;
    for (int m_i = _EU1D_NVARS_; m_i < nvars; m_i++) {
      f[p*nvars + m_i] = v * u[p*nvars + m_i];
    }
  }
}

/* Launch wrapper */
extern "C" {
void gpu_launch_euler1d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  int total_points = 1;
  for (int i = 0; i < ndims; i++) total_points *= (dim[i] + 2 * ghosts);
  int gridSize = (total_points + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_euler1d_flux_kernel, gridSize, blockSize)(
    f, u, nvars, ndims, dim_gpu, stride_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu);
  GPUFree(stride_gpu);
#endif
}
} /* extern "C" */

