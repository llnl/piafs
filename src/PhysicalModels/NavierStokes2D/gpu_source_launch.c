/*! @file gpu_source_launch.c
    @brief GPU source kernel launch wrappers (CPU fallback)
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_source.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_ns2d_source_zero(double *source, int nvars, int npoints, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < npoints; i++) {
    for (int v = 0; v < nvars; v++) {
      source[i*nvars + v] = 0.0;
    }
  }
#else
  (void)source; (void)nvars; (void)npoints; (void)blockSize;
#endif
}

