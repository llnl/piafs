/*! @file gpu_interpolation_launch.c
    @brief GPU interpolation kernel launch wrappers
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_interpolation.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_weno5_interpolation(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ninterfaces, int stride, int upw, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified version */
  static const double one_sixth = 1.0/6.0;
  for (int i = 0; i < ninterfaces; i++) {
    int qm1, qm2, qm3, qp1, qp2;
    if (upw > 0) {
      qm1 = i - 1 + stride;
      qm3 = qm1 - 2*stride;
      qm2 = qm1 - stride;
      qp1 = qm1 + stride;
      qp2 = qm1 + 2*stride;
    } else {
      qm1 = i + stride;
      qm3 = qm1 + 2*stride;
      qm2 = qm1 + stride;
      qp1 = qm1 - stride;
      qp2 = qm1 - 2*stride;
    }
    for (int v = 0; v < nvars; v++) {
      double f1 = (2*one_sixth)*fC[qm3*nvars+v] + (-7*one_sixth)*fC[qm2*nvars+v] + (11*one_sixth)*fC[qm1*nvars+v];
      double f2 = (-one_sixth)*fC[qm2*nvars+v] + (5*one_sixth)*fC[qm1*nvars+v] + (2*one_sixth)*fC[qp1*nvars+v];
      double f3 = (2*one_sixth)*fC[qm1*nvars+v] + (5*one_sixth)*fC[qp1*nvars+v] + (-one_sixth)*fC[qp2*nvars+v];
      fI[i*nvars+v] = w1[i*nvars+v]*f1 + w2[i*nvars+v]*f2 + w3[i*nvars+v]*f3;
    }
  }
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
  for (int i = 0; i < ninterfaces; i++) {
    int q1 = i + stride;
    int q2 = i + stride + 1;
    for (int v = 0; v < nvars; v++) {
      fI[i*nvars+v] = 0.5 * (fC[q1*nvars+v] + fC[q2*nvars+v]);
    }
  }
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
  static const double c0 = -1.0/12.0, c1 = 7.0/12.0, c2 = 7.0/12.0, c3 = -1.0/12.0;
  for (int i = 0; i < ninterfaces; i++) {
    int qm1 = i - 1 + stride;
    int q0 = i + stride;
    int qp1 = i + 1 + stride;
    int qp2 = i + 2 + stride;
    for (int v = 0; v < nvars; v++) {
      fI[i*nvars+v] = c0*fC[qm1*nvars+v] + c1*fC[q0*nvars+v] + c2*fC[qp1*nvars+v] + c3*fC[qp2*nvars+v];
    }
  }
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
  static const double one_third = 1.0/3.0, one_sixth = 1.0/6.0;
  for (int i = 0; i < ninterfaces; i++) {
    int qm2, qm1, q0;
    if (upw > 0) {
      qm2 = i - 2 + stride;
      qm1 = i - 1 + stride;
      q0 = i + stride;
    } else {
      qm2 = i + 2 + stride;
      qm1 = i + 1 + stride;
      q0 = i + stride;
    }
    for (int v = 0; v < nvars; v++) {
      double df1 = fC[qm1*nvars+v] - fC[qm2*nvars+v];
      double df2 = fC[q0*nvars+v] - fC[qm1*nvars+v];
      double num = 3.0 * df1 * df2 + eps;
      double den = 2.0 * (df2 - df1) * (df2 - df1) + 3.0 * df1 * df2 + eps;
      double phi = (den > 1e-14) ? num / den : 1.0;
      fI[i*nvars+v] = fC[qm1*nvars+v] + phi * (one_third*df2 + one_sixth*df1);
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (ninterfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_kernel, gridSize, blockSize)(
    fI, fC, nvars, ninterfaces, stride, upw, eps
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

