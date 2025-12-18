/*! @file gpu_cfl_launch.c
    @brief CPU fallback for GPU Euler 1D CFL launch
*/

int gpu_launch_euler1d_compute_cfl(
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

