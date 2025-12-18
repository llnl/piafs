/*! @file gpu_interpolation_launch.cu
    @brief GPU interpolation kernel launch wrappers (compiled with CUDA/HIP)
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_interpolation.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

extern "C" {

void gpu_launch_weno5_interpolation(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ninterfaces, int stride, int upw, int blockSize
)
{
#ifdef GPU_NONE
  /* Should not happen in GPU builds, but keep a safe fallback. */
  (void)fI; (void)fC; (void)w1; (void)w2; (void)w3;
  (void)nvars; (void)ninterfaces; (void)stride; (void)upw; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (ninterfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_weno5_interpolation_kernel, gridSize, blockSize)(
    fI, fC, w1, w2, w3, nvars, ninterfaces, stride, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_central2_interpolation(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ninterfaces; (void)stride; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (ninterfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_central2_interpolation_kernel, gridSize, blockSize)(
    fI, fC, nvars, ninterfaces, stride
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_central4_interpolation(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ninterfaces; (void)stride; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (ninterfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_central4_interpolation_kernel, gridSize, blockSize)(
    fI, fC, nvars, ninterfaces, stride
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_muscl3_interpolation(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int upw, double eps, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ninterfaces; (void)stride; (void)upw; (void)eps; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (ninterfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_kernel, gridSize, blockSize)(
    fI, fC, nvars, ninterfaces, stride, upw, eps
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

} /* extern "C" */


