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

