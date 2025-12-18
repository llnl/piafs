/*! @file gpu_hyperbolic_launch.c
    @brief GPU hyperbolic kernel launch wrappers
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_hyperbolic.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_hyperbolic_flux_derivative(
  double *hyp, const double *fluxI, const double *dxinv,
  int nvars, int npoints, int dir_offset, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
  for (int i = 0; i < npoints; i++) {
    double dx = dxinv[dir_offset + i];
    for (int v = 0; v < nvars; v++) {
      if (i + 1 < npoints + 1) {
        hyp[i*nvars + v] += dx * (fluxI[(i+1)*nvars + v] - fluxI[i*nvars + v]);
      }
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_hyperbolic_flux_derivative_kernel, gridSize, blockSize)(
    hyp, fluxI, dxinv, nvars, npoints, dir_offset
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_hyperbolic_flux_derivative_nd(
  double *hyp, const double *fluxI, const double *dxinv,
  double *StageBoundaryIntegral,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int dir_offset, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified version */
  int npoints_dir = dim[dir];
  int nlines = 1;
  for (int i = 0; i < ndims; i++) {
    if (i != dir) nlines *= dim[i];
  }
  int total_points = nlines * npoints_dir;
  for (int idx = 0; idx < total_points; idx++) {
    int line_idx = idx / npoints_dir;
    int point_idx = idx % npoints_dir;
    /* Simplified CPU version - would need full index computation */
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int npoints_dir = dim[dir];
  int nlines = 1;
  for (int i = 0; i < ndims; i++) {
    if (i != dir) nlines *= dim[i];
  }
  int total_points = nlines * npoints_dir;
  int gridSize = (total_points + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_hyperbolic_flux_derivative_nd_kernel, gridSize, blockSize)(
    hyp, fluxI, dxinv, StageBoundaryIntegral,
    nvars, ndims, dim, stride_with_ghosts, ghosts, dir, dir_offset
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_default_upwinding(
  double *fI, const double *fL, const double *fR,
  int nvars, int ninterfaces, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
  for (int i = 0; i < ninterfaces; i++) {
    for (int v = 0; v < nvars; v++) {
      fI[i*nvars + v] = 0.5 * (fL[i*nvars + v] + fR[i*nvars + v]);
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (ninterfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_default_upwinding_kernel, gridSize, blockSize)(
    fI, fL, fR, nvars, ninterfaces
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

