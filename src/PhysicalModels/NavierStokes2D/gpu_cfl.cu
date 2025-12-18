/*! @file gpu_cfl.cu
    @brief GPU kernel for Navier-Stokes 2D CFL computation
*/

#include <gpu.h>
#include <physicalmodels/navierstokes2d.h>
#include <math.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Compute local CFL for each grid point (NavierStokes2D) */
GPU_KERNEL void gpu_ns2d_compute_cfl_kernel(
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
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    int index[3];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      index[i] = temp % dim[i];
      temp /= dim[i];
    }
    
    int p = 0;
    for (int i = 0; i < ndims; i++) {
      p += (index[i] + ghosts) * stride_with_ghosts[i];
    }
    
    /* Get flow variables (GPU-safe) */
    double rho = u[p*nvars + 0];
    double vx = (rho == 0.0) ? 0.0 : u[p*nvars + 1] / rho;
    double vy = (rho == 0.0) ? 0.0 : u[p*nvars + 2] / rho;
    double e = u[p*nvars + 3];
    double vsq = (vx*vx) + (vy*vy);
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
    
    double max_local_cfl = 0.0;
    for (int dir = 0; dir < ndims; dir++) {
      int dir_offset = 0;
      for (int k = 0; k < dir; k++) {
        dir_offset += (dim[k] + 2 * ghosts);
      }
      double dxinv_dir = dxinv[dir_offset + ghosts + index[dir]];
      
      double v_mag = 0.0;
      if (dir == 0) v_mag = fabs(vx);
      else if (dir == 1) v_mag = fabs(vy);
      
      double local_cfl = (v_mag + c) * dt * dxinv_dir;
      if (isnan(local_cfl) || isinf(local_cfl)) {
        local_cfl = 0.0;
      }
      if (local_cfl > max_local_cfl) {
        max_local_cfl = local_cfl;
      }
    }
    
    cfl_local[idx] = max_local_cfl;
  }
}

