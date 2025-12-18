/*! @file gpu_cfl_launch.cu
    @brief Launch wrapper for GPU Navier-Stokes 3D CFL kernel
*/

#include <gpu.h>
#include <gpu_cfl.h>
#include <hypar.h>
#include <physicalmodels/navierstokes3d.h>

extern "C" {

/* Launch GPU kernel to compute local CFL values */
int gpu_launch_ns3d_compute_cfl(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  void *s,
  double dt
)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  
  int nvars = solver->nvars;
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  /* Compute total number of grid points (without ghosts) */
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }
  
  if (npoints == 0) {
    return 0;
  }
  
  /* Copy dim and stride_with_ghosts to GPU */
  int *dim_gpu = NULL;
  int *stride_gpu = NULL;
  int size_dim = ndims * sizeof(int);
  int size_stride = ndims * sizeof(int);
  
  GPU_MALLOC((void**)&dim_gpu, size_dim);
  GPU_MALLOC((void**)&stride_gpu, size_stride);
  
  GPU_MEMCPY(dim_gpu, dim, size_dim, GPU_MEMCPY_H2D);
  GPU_MEMCPY(stride_gpu, stride_with_ghosts, size_stride, GPU_MEMCPY_H2D);
  GPU_DEVICE_SYNC();
  
  /* Configure grid and block */
  int blockSize = 256;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  
  /* Launch kernel */
  gpu_ns3d_compute_cfl_kernel<<<gridSize, blockSize>>>(
    u,
    dxinv,
    cfl_local,
    nvars,
    ndims,
    dim_gpu,
    stride_gpu,
    ghosts,
    dt,
    param->gamma
  );
  
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPU_DEVICE_SYNC();
  
  /* Free temporary GPU arrays */
  GPU_FREE(dim_gpu);
  GPU_FREE(stride_gpu);
  
  return 0;
}

} /* extern "C" */

