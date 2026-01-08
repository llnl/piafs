/*! @file gpu_upwind_launch.cu
    @brief GPU upwind kernel launch wrappers for NavierStokes3D with memory pooling
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_upwind.h>

/* Kernel declarations */
extern GPU_KERNEL void gpu_ns3d_upwind_roe_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

/* Specialized RF kernels */
extern GPU_KERNEL void gpu_ns3d_upwind_rf_kernel_nvars5(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

extern GPU_KERNEL void gpu_ns3d_upwind_rf_kernel_nvars12(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

extern GPU_KERNEL void gpu_ns3d_upwind_rf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

/* Specialized LLF kernels */
extern GPU_KERNEL void gpu_ns3d_upwind_llf_kernel_nvars5(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

extern GPU_KERNEL void gpu_ns3d_upwind_llf_kernel_nvars12(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

extern GPU_KERNEL void gpu_ns3d_upwind_llf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

extern GPU_KERNEL void gpu_ns3d_upwind_rusanov_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
);

/* ============================================================================
   Memory pool for workspace arrays
   Eliminates repeated cudaMalloc/cudaFree overhead (was 55% of CUDA API time)
   ============================================================================ */

/* Static device buffers for dimension arrays */
static int *d_dim = NULL;
static int *d_stride_with_ghosts = NULL;
static int *d_bounds_inter = NULL;
static int d_arrays_capacity = 0;

/* Workspace memory pool */
static double *d_workspace_pool = NULL;
static size_t d_workspace_capacity = 0;

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

/* Get workspace from pool, growing if necessary */
static double* get_workspace(size_t required_size) {
  if (required_size <= d_workspace_capacity) {
    return d_workspace_pool;
  }

  /* Need to grow the pool - free old and allocate new */
  if (d_workspace_pool) {
    GPUFree(d_workspace_pool);
    d_workspace_pool = NULL;
  }

  /* Allocate with 20% extra to reduce future reallocations */
  size_t new_capacity = required_size + required_size / 5;
  if (GPUAllocate((void**)&d_workspace_pool, new_capacity * sizeof(double))) {
    d_workspace_capacity = 0;
    return NULL;
  }

  d_workspace_capacity = new_capacity;
  return d_workspace_pool;
}

extern "C" {

void gpu_launch_ns3d_upwind_roe(
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
  int total_threads = numBlocks * blockSize;

  /* Get workspace from pool: 3*nvars + 5*nvars*nvars per thread */
  size_t workspace_per_thread = 3 * nvars + 5 * nvars * nvars;
  size_t total_workspace = total_threads * workspace_per_thread;
  double *workspace = get_workspace(total_workspace);
  if (!workspace) {
    fprintf(stderr, "Error: Failed to get workspace for NS3D Roe upwinding\n");
    return;
  }

  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_roe_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma, workspace
  );
  /* No GPUFree - workspace stays in pool for reuse */
}

void gpu_launch_ns3d_upwind_rf(
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
  int total_threads = numBlocks * blockSize;

  /* Dispatch to specialized kernels for common nvars values */
  if (nvars == 5) {
    /* nvars=5 uses register-based workspace - no allocation needed */
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel_nvars5, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, NULL
    );
  } else if (nvars == 12) {
    /* nvars=12 uses register-based workspace (5x5 eigenvectors only) - no allocation needed */
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel_nvars12, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, NULL
    );
  } else {
    /* General fallback: get workspace from pool */
    size_t workspace_per_thread = 9 * nvars + 2 * nvars * nvars;
    size_t total_workspace = total_threads * workspace_per_thread;
    double *workspace = get_workspace(total_workspace);
    if (!workspace) {
      fprintf(stderr, "Error: Failed to get workspace for NS3D RF upwinding (nvars=%d)\n", nvars);
      return;
    }
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, workspace
    );
    /* No GPUFree - workspace stays in pool for reuse */
  }
}

void gpu_launch_ns3d_upwind_llf(
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
  int total_threads = numBlocks * blockSize;

  /* Dispatch to specialized kernels for common nvars values */
  if (nvars == 5) {
    /* nvars=5 uses register-based workspace - no allocation needed */
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_llf_kernel_nvars5, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, NULL
    );
  } else if (nvars == 12) {
    /* nvars=12 uses register-based workspace (5x5 eigenvectors only) - no allocation needed */
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_llf_kernel_nvars12, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, NULL
    );
  } else {
    /* General fallback: get workspace from pool */
    size_t workspace_per_thread = 9 * nvars + 3 * nvars * nvars;
    size_t total_workspace = total_threads * workspace_per_thread;
    double *workspace = get_workspace(total_workspace);
    if (!workspace) {
      fprintf(stderr, "Error: Failed to get workspace for NS3D LLF upwinding (nvars=%d)\n", nvars);
      return;
    }
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_llf_kernel, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, workspace
    );
    /* No GPUFree - workspace stays in pool for reuse */
  }
}

void gpu_launch_ns3d_upwind_rusanov(
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
  int total_threads = numBlocks * blockSize;

  /* Get workspace from pool: 2*nvars per thread */
  size_t workspace_per_thread = 2 * nvars;
  size_t total_workspace = total_threads * workspace_per_thread;
  double *workspace = get_workspace(total_workspace);
  if (!workspace) {
    fprintf(stderr, "Error: Failed to get workspace for NS3D Rusanov upwinding\n");
    return;
  }

  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rusanov_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma, workspace
  );
  /* No GPUFree - workspace stays in pool for reuse */
}

} /* extern "C" */
