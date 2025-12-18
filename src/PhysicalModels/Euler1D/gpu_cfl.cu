/*! @file gpu_cfl.cu
    @brief GPU kernel for Euler 1D CFL computation
*/

#include <gpu.h>
#include <physicalmodels/euler1d.h>
#include <math.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Compute local CFL for each grid point (Euler1D) */
GPU_KERNEL void gpu_euler1d_compute_cfl_kernel(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  int nvars,
  int ndims,
  const int *dim,
  const int *stride_with_ghosts,
  int ghosts,
  double dt,
  double gamma
)
{
  /* CFL is computed over interior points only (no ghosts) */
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    /* Decode idx into multi-dimensional interior index */
    int index[3] = {0,0,0};
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      index[i] = temp % dim[i];
      temp /= dim[i];
    }
    
    /* Compute 1D index into solution array (which includes ghosts) */
    /* For 1D: p = index[0] + ghosts */
    int p = index[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) {
      p = p * (dim[i] + 2*ghosts) + (index[i] + ghosts);
    }
    
    /* Get flow variables (GPU-safe) */
    double rho = u[p*nvars + 0];
    double v = (rho == 0.0) ? 0.0 : u[p*nvars + 1] / rho;
    double e = u[p*nvars + 2];
    double vsq = v * v;
    double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    
    if (isnan(rho) || isinf(rho) || rho <= 0.0 || 
        isnan(P) || isinf(P) || P <= 0.0) {
      cfl_local[idx] = 0.0;
      return;
    }
    
    double c = sqrt(gamma * P / rho);
    if (isnan(c) || isinf(c) || c <= 0.0) {
      cfl_local[idx] = 0.0;
      return;
    }
    
    /* Get dxinv for direction 0 */
    int dir_offset = 0;
    double dxinv_dir = dxinv[dir_offset + ghosts + index[0]];
    
    double local_cfl = (fabs(v) + c) * dt * dxinv_dir;
    if (isnan(local_cfl) || isinf(local_cfl)) {
      local_cfl = 0.0;
    }
    
    cfl_local[idx] = local_cfl;
  }
}

