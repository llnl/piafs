/*! @file gpu_upwind_launch.cu
    @brief GPU upwind kernel launch wrappers for Euler1D
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_upwind.h>

/* Kernel declarations */
extern GPU_KERNEL void gpu_euler1d_upwind_roe_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
);

extern GPU_KERNEL void gpu_euler1d_upwind_rf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
);

extern GPU_KERNEL void gpu_euler1d_upwind_llf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
);

extern GPU_KERNEL void gpu_euler1d_upwind_rusanov_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
);

/* Static device buffers for dimension arrays */
static int *d_dim = NULL;
static int *d_stride_with_ghosts = NULL;
static int *d_bounds_inter = NULL;
static int d_arrays_capacity = 0;

static int ensure_device_arrays(int ndims) {
  if (ndims <= d_arrays_capacity) return 0;

  /* Free old buffers */
  if (d_dim) { GPUFree(d_dim); d_dim = NULL; }
  if (d_stride_with_ghosts) { GPUFree(d_stride_with_ghosts); d_stride_with_ghosts = NULL; }
  if (d_bounds_inter) { GPUFree(d_bounds_inter); d_bounds_inter = NULL; }

  /* Allocate new buffers */
  if (GPUAllocate((void**)&d_dim, ndims * sizeof(int))) return 1;
  if (GPUAllocate((void**)&d_stride_with_ghosts, ndims * sizeof(int))) {
    GPUFree(d_dim); d_dim = NULL;
    return 1;
  }
  if (GPUAllocate((void**)&d_bounds_inter, ndims * sizeof(int))) {
    GPUFree(d_dim); d_dim = NULL;
    GPUFree(d_stride_with_ghosts); d_stride_with_ghosts = NULL;
    return 1;
  }

  d_arrays_capacity = ndims;
  return 0;
}

extern "C" {

void gpu_launch_euler1d_upwind_roe(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
  /* Ensure device arrays are allocated */
  if (ensure_device_arrays(ndims)) return;

  /* Copy dimension arrays to device */
  GPUCopyToDevice(d_dim, dim, ndims * sizeof(int));
  GPUCopyToDevice(d_stride_with_ghosts, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(d_bounds_inter, bounds_inter, ndims * sizeof(int));

  /* Calculate total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  /* Launch kernel */
  int numBlocks = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_euler1d_upwind_roe_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma
  );
}

void gpu_launch_euler1d_upwind_rf(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
  if (ensure_device_arrays(ndims)) return;
  GPUCopyToDevice(d_dim, dim, ndims * sizeof(int));
  GPUCopyToDevice(d_stride_with_ghosts, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(d_bounds_inter, bounds_inter, ndims * sizeof(int));

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int numBlocks = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_euler1d_upwind_rf_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma
  );
}

void gpu_launch_euler1d_upwind_llf(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
  if (ensure_device_arrays(ndims)) return;
  GPUCopyToDevice(d_dim, dim, ndims * sizeof(int));
  GPUCopyToDevice(d_stride_with_ghosts, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(d_bounds_inter, bounds_inter, ndims * sizeof(int));

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int numBlocks = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_euler1d_upwind_llf_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma
  );
}

void gpu_launch_euler1d_upwind_rusanov(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
  if (ensure_device_arrays(ndims)) return;
  GPUCopyToDevice(d_dim, dim, ndims * sizeof(int));
  GPUCopyToDevice(d_stride_with_ghosts, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(d_bounds_inter, bounds_inter, ndims * sizeof(int));

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int numBlocks = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_euler1d_upwind_rusanov_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma
  );
}

} /* extern "C" */
