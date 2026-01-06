/*! @file gpu_upwind_launch.cu
    @brief GPU upwind kernel launch wrappers for NavierStokes3D with dynamic workspace allocation
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
  
  /* Allocate workspace for Roe scheme: 3*nvars + 5*nvars*nvars per thread */
  size_t workspace_per_thread = 3 * nvars + 5 * nvars * nvars;
  size_t total_workspace = total_threads * workspace_per_thread;
  double *workspace = NULL;
  if (GPUAllocate((void**)&workspace, total_workspace * sizeof(double))) {
    fprintf(stderr, "Error: Failed to allocate workspace for NS3D Roe upwinding\n");
    return;
  }

  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_roe_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma, workspace
  );
  
  GPUFree(workspace);
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
    double *workspace = NULL;  /* Not used for nvars=5 */
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel_nvars5, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, workspace
    );
  } else if (nvars == 12) {
    /* Allocate workspace for nvars=12: 9*nvars + 2*nvars*nvars per thread */
    size_t workspace_per_thread = 9 * nvars + 2 * nvars * nvars;
    size_t total_workspace = total_threads * workspace_per_thread;
    double *workspace = NULL;
    if (GPUAllocate((void**)&workspace, total_workspace * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate workspace for NS3D RF upwinding (nvars=12)\n");
      return;
    }
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel_nvars12, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, workspace
    );
    GPUFree(workspace);
  } else {
    /* General fallback for other nvars: 9*nvars + 2*nvars*nvars per thread */
    size_t workspace_per_thread = 9 * nvars + 2 * nvars * nvars;
    size_t total_workspace = total_threads * workspace_per_thread;
    double *workspace = NULL;
    if (GPUAllocate((void**)&workspace, total_workspace * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate workspace for NS3D RF upwinding (nvars=%d)\n", nvars);
      return;
    }
    GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel, numBlocks, blockSize)(
      fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
      ghosts, dir, gamma, workspace
    );
    GPUFree(workspace);
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
  
  /* Allocate workspace for LLF scheme: 9*nvars + 3*nvars*nvars per thread */
  size_t workspace_per_thread = 9 * nvars + 3 * nvars * nvars;
  size_t total_workspace = total_threads * workspace_per_thread;
  double *workspace = NULL;
  if (GPUAllocate((void**)&workspace, total_workspace * sizeof(double))) {
    fprintf(stderr, "Error: Failed to allocate workspace for NS3D LLF upwinding\n");
    return;
  }

  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_llf_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma, workspace
  );
  
  GPUFree(workspace);
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
  
  /* Allocate workspace for Rusanov scheme: 2*nvars per thread */
  size_t workspace_per_thread = 2 * nvars;
  size_t total_workspace = total_threads * workspace_per_thread;
  double *workspace = NULL;
  if (GPUAllocate((void**)&workspace, total_workspace * sizeof(double))) {
    fprintf(stderr, "Error: Failed to allocate workspace for NS3D Rusanov upwinding\n");
    return;
  }

  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rusanov_kernel, numBlocks, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, d_dim, d_stride_with_ghosts, d_bounds_inter,
    ghosts, dir, gamma, workspace
  );
  
  GPUFree(workspace);
}

} /* extern "C" */
