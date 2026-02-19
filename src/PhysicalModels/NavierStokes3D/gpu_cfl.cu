/*! @file gpu_cfl.cu
    @brief GPU kernel for Navier-Stokes 3D CFL computation
*/

#include <gpu.h>
#include <physicalmodels/navierstokes3d.h>
#include <math.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Compute local CFL for each grid point (NavierStokes3D)
   Computes local CFL = (|v| + c) * dt * dxinv for each direction
   and stores the maximum local CFL in cfl_local array
*/
GPU_KERNEL void gpu_ns3d_compute_cfl_kernel(
  const double *u,              /* input: solution array */
  const double *dxinv,          /* input: 1/dx array */
  double *cfl_local,            /* output: local CFL for each grid point */
  int nvars,                    /* number of variables */
  int ndims,                    /* number of dimensions */
  const int *dim,               /* dimension sizes (without ghosts) */
  const int *stride_with_ghosts, /* stride array */
  int ghosts,                   /* number of ghost points */
  double dt,                    /* time step */
  double gamma                  /* ratio of specific heats */
)
{
  /* Compute total number of grid points (without ghosts) */
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    /* Decompose idx into multi-dimensional index */
    int index[3]; /* Support up to 3D */
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      index[i] = temp % dim[i];
      temp /= dim[i];
    }

    /* Compute 1D array index for this point (with ghosts) */
    int p = 0;
    for (int i = 0; i < ndims; i++) {
      p += (index[i] + ghosts) * stride_with_ghosts[i];
    }

    /* Get flow variables (GPU-safe version without fprintf/exit) */
    double rho = u[p*nvars + 0];
    double vx = (rho == 0.0) ? 0.0 : u[p*nvars + 1] / rho;
    double vy = (rho == 0.0) ? 0.0 : u[p*nvars + 2] / rho;
    double vz = (rho == 0.0) ? 0.0 : u[p*nvars + 3] / rho;
    double e = u[p*nvars + 4];
    double vsq = (vx*vx) + (vy*vy) + (vz*vz);
    double P = (e - 0.5*rho*vsq) * (gamma - 1.0);

    /* Check for invalid values */
    if (isnan(rho) || isinf(rho) || rho <= 0.0 ||
        isnan(P) || isinf(P) || P <= 0.0) {
      cfl_local[idx] = 0.0; /* Skip invalid points */
      return;
    }

    /* Compute speed of sound */
    double c = sqrt(gamma * P / rho);
    if (isnan(c) || isinf(c) || c <= 0.0) {
      cfl_local[idx] = 0.0; /* Skip invalid points */
      return;
    }

    /* Compute local CFL for each direction */
    double max_local_cfl = 0.0;
    for (int dir = 0; dir < ndims; dir++) {
      /* Get dxinv for this direction */
      int dir_offset = 0;
      for (int k = 0; k < dir; k++) {
        dir_offset += (dim[k] + 2 * ghosts);
      }
      double dxinv_dir = dxinv[dir_offset + ghosts + index[dir]];

      /* Compute velocity magnitude in this direction */
      double v_mag = 0.0;
      if (dir == 0) v_mag = fabs(vx);
      else if (dir == 1) v_mag = fabs(vy);
      else if (dir == 2) v_mag = fabs(vz);

      /* Compute local CFL */
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

