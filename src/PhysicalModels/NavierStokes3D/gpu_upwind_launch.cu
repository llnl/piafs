/*! @file gpu_upwind_launch.cu
    @brief GPU upwind kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_upwind.h>

#define DEFAULT_BLOCK_SIZE 256

extern "C" {
void gpu_launch_ns3d_upwind_roe(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified version */
  /* Would need full CPU implementation here */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  
  /* Copy dim, stride_with_ghosts, and bounds_inter to GPU if needed */
  int *dim_gpu = NULL;
  int *stride_gpu = NULL;
  int *bounds_inter_gpu = NULL;
  
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate dim_gpu for upwind\n");
    return;
  }
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate stride_gpu for upwind\n");
    GPUFree(dim_gpu);
    return;
  }
  if (GPUAllocate((void**)&bounds_inter_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate bounds_inter_gpu for upwind\n");
    GPUFree(dim_gpu);
    GPUFree(stride_gpu);
    return;
  }
  
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(bounds_inter_gpu, bounds_inter, ndims * sizeof(int));
  
  /* Compute total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  
  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_roe_kernel, gridSize, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, dim_gpu, stride_gpu, bounds_inter_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  
  GPUFree(dim_gpu);
  GPUFree(stride_gpu);
  GPUFree(bounds_inter_gpu);
#endif
}

void gpu_launch_ns3d_upwind_rf(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_inter_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_inter_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(bounds_inter_gpu, bounds_inter, ndims * sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rf_kernel, gridSize, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, dim_gpu, stride_gpu, bounds_inter_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_inter_gpu);
#endif
}

void gpu_launch_ns3d_upwind_llf(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_inter_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_inter_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(bounds_inter_gpu, bounds_inter, ndims * sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_llf_kernel, gridSize, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, dim_gpu, stride_gpu, bounds_inter_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_inter_gpu);
#endif
}

void gpu_launch_ns3d_upwind_rusanov(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_inter_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_inter_gpu, ndims * sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(bounds_inter_gpu, bounds_inter, ndims * sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_upwind_rusanov_kernel, gridSize, blockSize)(
    fI, fL, fR, uL, uR, u, nvars, ndims, dim_gpu, stride_gpu, bounds_inter_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_inter_gpu);
#endif
}
} /* extern "C" */

