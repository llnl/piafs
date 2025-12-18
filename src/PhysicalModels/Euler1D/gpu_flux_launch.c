/*! @file gpu_flux_launch.c
    @brief GPU flux kernel launch wrappers (CPU fallback) for Euler1D
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_flux.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_euler1d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)f; (void)u; (void)nvars; (void)ndims; (void)dim;
  (void)stride_with_ghosts; (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  (void)f; (void)u; (void)nvars; (void)ndims; (void)dim;
  (void)stride_with_ghosts; (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#endif
}

