/*! @file gpu_cfl_launch.c
    @brief CPU fallback for GPU Navier-Stokes 2D CFL launch
*/

int gpu_launch_ns2d_compute_cfl(
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
  return 1;
}

