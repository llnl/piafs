/*! @file gpu_flux_launch.cu
    @brief GPU flux kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_flux.h>

#define DEFAULT_BLOCK_SIZE 256

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
  
  /* Copy dim and stride_with_ghosts to GPU if needed */
  int *dim_gpu = NULL;
  int *stride_gpu = NULL;
  
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate dim_gpu for flux\n");
    return;
  }
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate stride_gpu for flux\n");
    GPUFree(dim_gpu);
    return;
  }
  
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  
  /* Compute total number of points */
  int total_points = 1;
  for (int i = 0; i < ndims; i++) {
    total_points *= (dim[i] + 2 * ghosts);
  }
  
  int gridSize = (total_points + blockSize - 1) / blockSize;
  
  GPU_KERNEL_LAUNCH(gpu_ns3d_flux_kernel, gridSize, blockSize)(
    f, u, nvars, ndims, dim_gpu, stride_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  
  GPUFree(dim_gpu);
  GPUFree(stride_gpu);
#endif
}
} /* extern "C" */

