/*! @file gpu_hyperbolic_launch.cu
    @brief GPU hyperbolic kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_runtime.h>
#ifndef GPU_NONE
#include <gpu_hyperbolic.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

extern "C" {

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

  /* dim and stride_with_ghosts are GPU pointers - need to copy to host to compute grid size */
  int *dim_host = (int*) malloc(ndims * sizeof(int));
  int *stride_host = (int*) malloc(ndims * sizeof(int));
  if (!dim_host || !stride_host) {
    fprintf(stderr, "Error: Failed to allocate host buffers for dim/stride in launch function\n");
    if (dim_host) free(dim_host);
    if (stride_host) free(stride_host);
    return;
  }

  GPUCopyToHost(dim_host, dim, ndims * sizeof(int));
  GPUCopyToHost(stride_host, stride_with_ghosts, ndims * sizeof(int));
  /* GPUCopyToHost uses a synchronous copy; avoid forced device sync here. */

  int npoints_dir = dim_host[dir];
  int nlines = 1;
  for (int i = 0; i < ndims; i++) {
    if (i != dir) nlines *= dim_host[i];
  }
  int total_points = nlines * npoints_dir;
  int gridSize = (total_points + blockSize - 1) / blockSize;

  /* Use specialized 3D kernels for nvars=5 or nvars=12 */
  if (ndims == 3 && nvars == 5) {
    GPU_KERNEL_LAUNCH(gpu_hyperbolic_flux_derivative_3d_nvars5_kernel, gridSize, blockSize)(
      hyp, fluxI, dxinv, StageBoundaryIntegral,
      dim_host[0], dim_host[1], dim_host[2],
      stride_host[0], stride_host[1], stride_host[2],
      ghosts, dir, dir_offset
    );
  } else if (ndims == 3 && nvars == 12) {
    GPU_KERNEL_LAUNCH(gpu_hyperbolic_flux_derivative_3d_nvars12_kernel, gridSize, blockSize)(
      hyp, fluxI, dxinv, StageBoundaryIntegral,
      dim_host[0], dim_host[1], dim_host[2],
      stride_host[0], stride_host[1], stride_host[2],
      ghosts, dir, dir_offset
    );
  } else {
    /* Fall back to generic N-D kernel */
    GPU_KERNEL_LAUNCH(gpu_hyperbolic_flux_derivative_nd_kernel, gridSize, blockSize)(
      hyp, fluxI, dxinv, StageBoundaryIntegral,
      nvars, ndims, dim, stride_with_ghosts, ghosts, dir, dir_offset
    );
  }
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());

  free(dim_host);
  free(stride_host);
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

} /* extern "C" */

