/*! @file gpu_flux_launch.c
    @brief GPU flux kernel launch wrappers (CPU fallback)
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_flux.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_ns3d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - would need full implementation */
  /* For now, just return - this should not be called when GPU_NONE */
  (void)f; (void)u; (void)nvars; (void)ndims; (void)dim;
  (void)stride_with_ghosts; (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  /* This should not be called - use .cu version */
  (void)f; (void)u; (void)nvars; (void)ndims; (void)dim;
  (void)stride_with_ghosts; (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#endif
}

