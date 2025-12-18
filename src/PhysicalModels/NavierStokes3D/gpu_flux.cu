/*! @file gpu_flux.cu
    @brief GPU kernels for NavierStokes3D flux computation
*/

#include <gpu.h>
#include <physicalmodels/navierstokes3d.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Compute flux for NavierStokes3D
   Each thread handles one grid point
*/
GPU_KERNEL void gpu_ns3d_flux_kernel(
  double *f,              /* output: flux array */
  const double *u,        /* input: solution array */
  int nvars,              /* number of variables */
  int ndims,              /* number of dimensions */
  const int *dim,         /* dimension sizes (without ghosts) */
  const int *stride_with_ghosts, /* stride array */
  int ghosts,             /* number of ghost points */
  int dir,                /* direction (0=x, 1=y, 2=z) */
  double gamma            /* gamma parameter */
)
{
  /* Compute total number of points (with ghosts) */
  int total_points = 1;
  for (int i = 0; i < ndims; i++) {
    total_points *= (dim[i] + 2 * ghosts);
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_points) {
    /* Decompose idx into multi-dimensional index */
    /* index[i] ranges from 0 to dim[i] + 2*ghosts - 1 (including ghost points) */
    int index[3]; /* Support up to 3D */
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      index[i] = temp % (dim[i] + 2 * ghosts);
      temp /= (dim[i] + 2 * ghosts);
    }
    
    /* Compute 1D index using stride_with_ghosts */
    /* stride_with_ghosts[i] is the stride for dimension i */
    /* index[i] already ranges from 0 to dim[i] + 2*ghosts - 1 (including ghosts) */
    /* So we use index[i] directly with stride_with_ghosts[i] */
    int p = 0;
    for (int i = 0; i < ndims; i++) {
      p += index[i] * stride_with_ghosts[i];
    }
    
    /* Get flow variables from solution */
    double rho = u[p*nvars + 0];
    double vx  = u[p*nvars + 1] / rho;
    double vy  = u[p*nvars + 2] / rho;
    double vz  = u[p*nvars + 3] / rho;
    double e   = u[p*nvars + 4];
    
    /* Compute pressure */
    double vsq = vx*vx + vy*vy + vz*vz;
    double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
    
    /* Compute flux based on direction */
    if (dir == _XDIR_) {
      f[p*nvars + 0] = rho * vx;
      f[p*nvars + 1] = rho * vx * vx + P;
      f[p*nvars + 2] = rho * vx * vy;
      f[p*nvars + 3] = rho * vx * vz;
      f[p*nvars + 4] = (e + P) * vx;
      for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
        f[p*nvars + m_i] = vx * u[p*nvars + m_i];
      }
    } else if (dir == _YDIR_) {
      f[p*nvars + 0] = rho * vy;
      f[p*nvars + 1] = rho * vy * vx;
      f[p*nvars + 2] = rho * vy * vy + P;
      f[p*nvars + 3] = rho * vy * vz;
      f[p*nvars + 4] = (e + P) * vy;
      for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
        f[p*nvars + m_i] = vy * u[p*nvars + m_i];
      }
    } else if (dir == _ZDIR_) {
      f[p*nvars + 0] = rho * vz;
      f[p*nvars + 1] = rho * vz * vx;
      f[p*nvars + 2] = rho * vz * vy;
      f[p*nvars + 3] = rho * vz * vz + P;
      f[p*nvars + 4] = (e + P) * vz;
      for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
        f[p*nvars + m_i] = vz * u[p*nvars + m_i];
      }
    }
  }
}

