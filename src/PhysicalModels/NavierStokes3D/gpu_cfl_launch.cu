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

  /* Compute total number of grid points (without ghosts) */
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }

  if (npoints == 0) {
    return 0;
  }

  /* Copy dim and stride_with_ghosts to GPU */
  /* Use cached metadata arrays - already on device */
  int *dim_gpu = solver->gpu_dim_local;
  int *stride_gpu = solver->gpu_stride_with_ghosts;

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

  /* No need to sync or free - using cached metadata */
  return 0;
}

} /* extern "C" */

