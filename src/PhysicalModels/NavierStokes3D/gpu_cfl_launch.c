/*! @file gpu_cfl_launch.c
    @brief CPU fallback for GPU Navier-Stokes 3D CFL launch
*/

#include <physicalmodels/navierstokes3d.h>
#include <hypar.h>

/* CPU fallback - just return error */
int gpu_launch_ns3d_compute_cfl(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  void *s,
  double dt
)
{
  (void) u;
  (void) dxinv;
  (void) cfl_local;
  (void) s;
  (void) dt;
  return 1; /* GPU not available */
}

