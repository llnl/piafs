/*! @file gpu_flux_launch.cu
    @brief GPU flux kernel launch wrappers with memory pooling
*/

#include <gpu.h>
#include <gpu_flux.h>

#define DEFAULT_BLOCK_SIZE 256

/* ============================================================================
   Static device buffers for dimension arrays - eliminates per-call allocation
   ============================================================================ */
static int *d_flux_dim = NULL;
static int *d_flux_stride = NULL;
static int d_flux_capacity = 0;

static int ensure_flux_device_arrays(int ndims) {
  if (ndims <= d_flux_capacity) return 0;

  /* Free old buffers */
  if (d_flux_dim) { GPUFree(d_flux_dim); d_flux_dim = NULL; }
  if (d_flux_stride) { GPUFree(d_flux_stride); d_flux_stride = NULL; }

  /* Allocate new buffers */
  if (GPUAllocate((void**)&d_flux_dim, ndims * sizeof(int))) return 1;
  if (GPUAllocate((void**)&d_flux_stride, ndims * sizeof(int))) {
    GPUFree(d_flux_dim); d_flux_dim = NULL;
    return 1;
  }

  d_flux_capacity = ndims;
  return 0;
}

extern "C" {
void gpu_launch_ns3d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified version */
  int total_points = 1;
  for (int i = 0; i < ndims; i++) {
    total_points *= (dim[i] + 2 * ghosts);
  }
  /* Would need full CPU implementation here */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  /* Use pooled device arrays */
  if (ensure_flux_device_arrays(ndims)) {
    fprintf(stderr, "Error: Failed to allocate device arrays for flux\n");
    return;
  }

  GPUCopyToDevice(d_flux_dim, dim, ndims * sizeof(int));
  GPUCopyToDevice(d_flux_stride, stride_with_ghosts, ndims * sizeof(int));

  /* Compute total number of points */
  int total_points = 1;
  for (int i = 0; i < ndims; i++) {
    total_points *= (dim[i] + 2 * ghosts);
  }

  int gridSize = (total_points + blockSize - 1) / blockSize;

  GPU_KERNEL_LAUNCH(gpu_ns3d_flux_kernel, gridSize, blockSize)(
    f, u, nvars, ndims, d_flux_dim, d_flux_stride, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No GPUFree - arrays stay in pool for reuse */
#endif
}
} /* extern "C" */

