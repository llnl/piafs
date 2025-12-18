/*! @file gpu_flux.cu
    @brief GPU kernels and launch wrappers for NavierStokes2D flux computation
*/

#include <gpu.h>
#include <gpu_flux.h>
#include <physicalmodels/navierstokes2d.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

#define DEFAULT_BLOCK_SIZE 256

/* Kernel: Compute flux for NavierStokes2D */
GPU_KERNEL void gpu_ns2d_flux_kernel(
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
    double vx = (rho == 0) ? 0 : u[p*nvars + 1] / rho;
    double vy = (rho == 0) ? 0 : u[p*nvars + 2] / rho;
    double e = u[p*nvars + 3];
    double vsq = vx*vx + vy*vy;
    double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
    
    if (dir == _XDIR_) {
      f[p*nvars + 0] = rho * vx;
      f[p*nvars + 1] = rho * vx * vx + P;
      f[p*nvars + 2] = rho * vx * vy;
      f[p*nvars + 3] = (e + P) * vx;
      for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
        f[p*nvars + m_i] = vx * u[p*nvars + m_i];
      }
    } else if (dir == _YDIR_) {
      f[p*nvars + 0] = rho * vy;
      f[p*nvars + 1] = rho * vy * vx;
      f[p*nvars + 2] = rho * vy * vy + P;
      f[p*nvars + 3] = (e + P) * vy;
      for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
        f[p*nvars + m_i] = vy * u[p*nvars + m_i];
      }
    }
  }
}

/* Launch wrapper */
extern "C" {
void gpu_launch_ns2d_flux(
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
  GPU_KERNEL_LAUNCH(gpu_ns2d_flux_kernel, gridSize, blockSize)(
    f, u, nvars, ndims, dim_gpu, stride_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu);
  GPUFree(stride_gpu);
#endif
}
} /* extern "C" */

