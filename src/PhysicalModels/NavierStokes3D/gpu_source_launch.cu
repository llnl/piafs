/*! @file gpu_source_launch.cu
    @brief GPU source kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_source.h>

#define DEFAULT_BLOCK_SIZE 256

extern "C" {
void gpu_launch_ns3d_source_zero(
  double *source, int nvars, int npoints, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
  for (int i = 0; i < npoints; i++) {
    for (int v = 0; v < nvars; v++) {
      source[i*nvars + v] = 0.0;
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int gridSize = (npoints + blockSize - 1) / blockSize;

  GPU_KERNEL_LAUNCH(gpu_ns3d_source_zero_kernel, gridSize, blockSize)(
    source, nvars, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}
} /* extern "C" */

