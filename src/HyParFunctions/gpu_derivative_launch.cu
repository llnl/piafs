/*! @file gpu_derivative_launch.cu
    @brief GPU derivative kernel launch wrappers
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_derivative.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

extern "C" {

void gpu_launch_first_derivative_second_order(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified */
  for (int i = ghosts; i < npoints - ghosts; i++) {
    int qC = i;
    int qL = i - stride;
    int qR = i + stride;
    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_derivative_second_order_kernel, gridSize, blockSize)(
    Df, f, nvars, npoints, ghosts, stride
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_first_derivative_fourth_order(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, int blockSize
)
{
#ifdef GPU_NONE
  static const double c0 = -1.0/12.0, c1 = 2.0/3.0, c2 = -2.0/3.0, c3 = 1.0/12.0;
  for (int i = ghosts; i < npoints - ghosts; i++) {
    int qC = i;
    int qL = i - stride;
    int qR = i + stride;
    int qLL = i - 2*stride;
    int qRR = i + 2*stride;
    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = c0*f[qLL*nvars+v] + c1*f[qL*nvars+v] + c2*f[qR*nvars+v] + c3*f[qRR*nvars+v];
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_derivative_fourth_order_kernel, gridSize, blockSize)(
    Df, f, nvars, npoints, ghosts, stride
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_first_derivative_first_order(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, double bias, int blockSize
)
{
#ifdef GPU_NONE
  for (int i = ghosts; i < npoints - ghosts; i++) {
    int qC = i;
    int qL = i - stride;
    int qR = i + stride;
    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = (bias > 0 ? f[qR*nvars+v] - f[qC*nvars+v] :
                       bias < 0 ? f[qC*nvars+v] - f[qL*nvars+v] :
                       0.5 * (f[qR*nvars+v] - f[qL*nvars+v]));
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_derivative_first_order_kernel, gridSize, blockSize)(
    Df, f, nvars, npoints, ghosts, stride, bias
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

} /* extern "C" */

