/*! @file gpu_hyperbolic.cu
    @brief GPU kernels for hyperbolic flux computation
*/

#include <gpu.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Compute hyperbolic flux derivative
   This kernel computes the derivative of the interface flux
   hyp[i] += dxinv * (fluxI[i+1] - fluxI[i])
*/
GPU_KERNEL void gpu_hyperbolic_flux_derivative_kernel(
  double *hyp,           /* output: hyperbolic term */
  const double *fluxI,   /* input: interface fluxes */
  const double *dxinv,   /* input: 1/dx array */
  int nvars,             /* number of variables */
  int npoints,           /* number of grid points */
  int dir_offset          /* offset in dxinv for current direction */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    /* Compute interface indices */
    int p = idx;
    int p1 = idx;  /* left interface */
    int p2 = idx + 1;  /* right interface (for interior points) */
    
    /* Get dxinv value for this point */
    double dx = dxinv[dir_offset + idx];
    
    for (int v = 0; v < nvars; v++) {
      if (p2 < npoints + 1) {  /* valid right interface */
        hyp[p*nvars + v] += dx * (fluxI[p2*nvars + v] - fluxI[p1*nvars + v]);
      }
    }
  }
}

/* Kernel: Default upwinding (arithmetic mean) */
GPU_KERNEL void gpu_default_upwinding_kernel(
  double *fI,   /* output: upwind interface flux */
  const double *fL,  /* input: left-biased flux */
  const double *fR,  /* input: right-biased flux */
  int nvars,    /* number of variables */
  int ninterfaces  /* number of interfaces */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ninterfaces) {
    for (int v = 0; v < nvars; v++) {
      fI[idx*nvars + v] = 0.5 * (fL[idx*nvars + v] + fR[idx*nvars + v]);
    }
  }
}

/* Kernel: Compute hyperbolic flux derivative for multi-dimensional arrays
   This kernel processes all grid points in parallel, computing the derivative
   along dimension dir for each point.
*/
GPU_KERNEL void gpu_hyperbolic_flux_derivative_nd_kernel(
  double *hyp,                    /* output: hyperbolic term */
  const double *fluxI,            /* input: interface fluxes */
  const double *dxinv,             /* input: 1/dx array */
  double *StageBoundaryIntegral,   /* output: boundary flux integrals */
  int nvars,                       /* number of variables */
  int ndims,                       /* number of dimensions */
  const int *dim,                  /* dimensions (without ghosts) */
  const int *stride_with_ghosts,   /* stride array */
  int ghosts,                      /* number of ghost points */
  int dir,                         /* current direction */
  int dir_offset                   /* offset in dxinv for current direction */
)
{
  /* Compute total number of grid points (without ghosts in direction dir) */
  int npoints_dir = dim[dir];
  int nlines = 1;
  for (int i = 0; i < ndims; i++) {
    if (i != dir) nlines *= dim[i];
  }
  int total_points = nlines * npoints_dir;
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_points) {
    /* Decompose idx into line index and point index along dir */
    int line_idx = idx / npoints_dir;
    int point_idx = idx % npoints_dir;
    
    /* Reconstruct multi-dimensional index for this point */
    int index[3]; /* Support up to 3D */
    int temp = line_idx;
    for (int i = ndims - 1; i >= 0; i--) {
      if (i == dir) continue;
      index[i] = temp % dim[i];
      temp /= dim[i];
    }
    index[dir] = point_idx;
    
    /* Compute 1D array index for this point (with ghosts) */
    int p = 0;
    for (int i = 0; i < ndims; i++) {
      p += (index[i] + ghosts) * stride_with_ghosts[i];
    }
    
    /* Compute interface indices - interfaces don't have ghost points */
    int index1[3], index2[3];
    for (int i = 0; i < ndims; i++) {
      index1[i] = index[i];
      index2[i] = index[i];
    }
    index2[dir]++;
    
    /* Interface array has dim_interface[d] = dim[d] + 1 points (no ghosts) */
    int dim_interface[3];
    for (int i = 0; i < ndims; i++) {
      dim_interface[i] = (i == dir) ? (dim[i] + 1) : dim[i];
    }

    /* Compute 1D interface indices (no ghosts) using the same ordering as _ArrayIndex1D_:
       index[0] is the fastest varying dimension. This must match the interpolation/upwind
       kernels which use:
         p = indexI[ndims-1]; for (i=ndims-2..0) p = p*bounds[i] + indexI[i];
    */
    int p1 = index1[ndims-1];
    int p2 = index2[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p1 = p1 * dim_interface[i] + index1[i];
      p2 = p2 * dim_interface[i] + index2[i];
    }
    
    /* Bounds check for interface indices */
    int size_interface = 1;
    for (int i = 0; i < ndims; i++) {
      size_interface *= dim_interface[i];
    }
    if (p1 < 0 || p1 >= size_interface || p2 < 0 || p2 >= size_interface) {
      /* Out of bounds - skip this point */
      return;
    }
    
    /* Get dxinv value for this point */
    double dx = dxinv[dir_offset + ghosts + point_idx];
    
    /* Validate dxinv */
    if (isnan(dx) || isinf(dx) || dx == 0.0) {
      /* Invalid dxinv - skip this point */
      return;
    }
    
    /* Compute derivative for all variables */
    for (int v = 0; v < nvars; v++) {
      double flux_diff = fluxI[p2 * nvars + v] - fluxI[p1 * nvars + v];
      if (isnan(flux_diff) || isinf(flux_diff)) {
        /* Invalid flux difference - skip this variable */
        continue;
      }
      hyp[p * nvars + v] += dx * flux_diff;
    }
    
    /* Handle boundary flux integrals (atomic operations for thread safety) */
    if (point_idx == 0) {
      for (int v = 0; v < nvars; v++) {
        /* Use atomic add for thread safety */
        #ifdef GPU_CUDA
        atomicAdd(&StageBoundaryIntegral[(2*dir+0)*nvars+v], -fluxI[p1*nvars+v]);
        #elif defined(GPU_HIP)
        atomicAdd(&StageBoundaryIntegral[(2*dir+0)*nvars+v], -fluxI[p1*nvars+v]);
        #endif
      }
    }
    if (point_idx == npoints_dir - 1) {
      for (int v = 0; v < nvars; v++) {
        #ifdef GPU_CUDA
        atomicAdd(&StageBoundaryIntegral[(2*dir+1)*nvars+v], fluxI[p2*nvars+v]);
        #elif defined(GPU_HIP)
        atomicAdd(&StageBoundaryIntegral[(2*dir+1)*nvars+v], fluxI[p2*nvars+v]);
        #endif
      }
    }
  }
}

/* Specialized 3D flux derivative kernel for nvars=5 (e.g., Euler/NS with 5 conserved variables) */
GPU_KERNEL void gpu_hyperbolic_flux_derivative_3d_nvars5_kernel(
  double *hyp,
  const double *fluxI,
  const double *dxinv,
  double *StageBoundaryIntegral,
  int ni, int nj, int nk,
  int stride_i, int stride_j, int stride_k,
  int ghosts,
  int dir,
  int dir_offset
)
{
  const int nvars = 5;
  int npoints_dir, dim_other1, dim_other2;

  if (dir == 0) {
    npoints_dir = ni; dim_other1 = nj; dim_other2 = nk;
  } else if (dir == 1) {
    npoints_dir = nj; dim_other1 = ni; dim_other2 = nk;
  } else {
    npoints_dir = nk; dim_other1 = ni; dim_other2 = nj;
  }

  int total_points = npoints_dir * dim_other1 * dim_other2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_points) return;

  /* Decompose idx into 3D indices */
  int point_idx = idx % npoints_dir;
  int tmp = idx / npoints_dir;
  int idx1 = tmp % dim_other1;
  int idx2 = tmp / dim_other1;

  int i, j, k;
  if (dir == 0) {
    i = point_idx; j = idx1; k = idx2;
  } else if (dir == 1) {
    j = point_idx; i = idx1; k = idx2;
  } else {
    k = point_idx; i = idx1; j = idx2;
  }

  /* Compute array index with ghosts */
  int p = (i + ghosts) * stride_i + (j + ghosts) * stride_j + (k + ghosts) * stride_k;

  /* Interface array dimensions (k stride not needed - it's the slowest varying) */
  int ni_I = (dir == 0) ? (ni + 1) : ni;
  int nj_I = (dir == 1) ? (nj + 1) : nj;

  /* Interface indices (using same ordering as _ArrayIndex1D_) */
  int i1 = i, j1 = j, k1 = k;
  int i2 = i, j2 = j, k2 = k;
  if (dir == 0) i2++; else if (dir == 1) j2++; else k2++;

  int p1 = k1 * (nj_I * ni_I) + j1 * ni_I + i1;
  int p2 = k2 * (nj_I * ni_I) + j2 * ni_I + i2;

  double dx = dxinv[dir_offset + ghosts + point_idx];

  /* Unrolled derivative computation */
  #pragma unroll
  for (int v = 0; v < 5; v++) {
    hyp[p * nvars + v] += dx * (fluxI[p2 * nvars + v] - fluxI[p1 * nvars + v]);
  }

  /* Boundary flux integrals */
  if (point_idx == 0) {
    #pragma unroll
    for (int v = 0; v < 5; v++) {
      atomicAdd(&StageBoundaryIntegral[(2*dir+0)*nvars+v], -fluxI[p1*nvars+v]);
    }
  }
  if (point_idx == npoints_dir - 1) {
    #pragma unroll
    for (int v = 0; v < 5; v++) {
      atomicAdd(&StageBoundaryIntegral[(2*dir+1)*nvars+v], fluxI[p2*nvars+v]);
    }
  }
}

/* Specialized 3D flux derivative kernel for nvars=12 (e.g., NS3D with chemistry) */
GPU_KERNEL void gpu_hyperbolic_flux_derivative_3d_nvars12_kernel(
  double *hyp,
  const double *fluxI,
  const double *dxinv,
  double *StageBoundaryIntegral,
  int ni, int nj, int nk,
  int stride_i, int stride_j, int stride_k,
  int ghosts,
  int dir,
  int dir_offset
)
{
  const int nvars = 12;
  int npoints_dir, dim_other1, dim_other2;

  if (dir == 0) {
    npoints_dir = ni; dim_other1 = nj; dim_other2 = nk;
  } else if (dir == 1) {
    npoints_dir = nj; dim_other1 = ni; dim_other2 = nk;
  } else {
    npoints_dir = nk; dim_other1 = ni; dim_other2 = nj;
  }

  int total_points = npoints_dir * dim_other1 * dim_other2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_points) return;

  /* Decompose idx into 3D indices */
  int point_idx = idx % npoints_dir;
  int tmp = idx / npoints_dir;
  int idx1 = tmp % dim_other1;
  int idx2 = tmp / dim_other1;

  int i, j, k;
  if (dir == 0) {
    i = point_idx; j = idx1; k = idx2;
  } else if (dir == 1) {
    j = point_idx; i = idx1; k = idx2;
  } else {
    k = point_idx; i = idx1; j = idx2;
  }

  /* Compute array index with ghosts */
  int p = (i + ghosts) * stride_i + (j + ghosts) * stride_j + (k + ghosts) * stride_k;

  /* Interface array dimensions (k stride not needed - it's the slowest varying) */
  int ni_I = (dir == 0) ? (ni + 1) : ni;
  int nj_I = (dir == 1) ? (nj + 1) : nj;

  /* Interface indices */
  int i1 = i, j1 = j, k1 = k;
  int i2 = i, j2 = j, k2 = k;
  if (dir == 0) i2++; else if (dir == 1) j2++; else k2++;

  int p1 = k1 * (nj_I * ni_I) + j1 * ni_I + i1;
  int p2 = k2 * (nj_I * ni_I) + j2 * ni_I + i2;

  double dx = dxinv[dir_offset + ghosts + point_idx];

  /* Unrolled derivative computation for 12 variables */
  #pragma unroll
  for (int v = 0; v < 12; v++) {
    hyp[p * nvars + v] += dx * (fluxI[p2 * nvars + v] - fluxI[p1 * nvars + v]);
  }

  /* Boundary flux integrals */
  if (point_idx == 0) {
    #pragma unroll
    for (int v = 0; v < 12; v++) {
      atomicAdd(&StageBoundaryIntegral[(2*dir+0)*nvars+v], -fluxI[p1*nvars+v]);
    }
  }
  if (point_idx == npoints_dir - 1) {
    #pragma unroll
    for (int v = 0; v < 12; v++) {
      atomicAdd(&StageBoundaryIntegral[(2*dir+1)*nvars+v], fluxI[p2*nvars+v]);
    }
  }
}

