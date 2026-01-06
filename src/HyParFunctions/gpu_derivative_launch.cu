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

/*******************************************************************************
 * 3D BATCHED DERIVATIVE LAUNCH WRAPPERS
 * These launch a single kernel to process the entire 3D domain.
 ******************************************************************************/

void gpu_launch_first_derivative_first_order_3d(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, double bias, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - only process interior (i,k) positions to match per-line behavior */
  int total = ni * nj * nk;
  for (int tid = 0; tid < total; tid++) {
    int i = tid % ni;
    int j = (tid / ni) % nj;
    int k = tid / (ni * nj);

    /* Skip ghost regions in non-derivative directions */
    if (dir == 0) {
      if (j < ghosts || j >= nj - ghosts || k < ghosts || k >= nk - ghosts) continue;
    } else if (dir == 1) {
      if (i < ghosts || i >= ni - ghosts || k < ghosts || k >= nk - ghosts) continue;
    } else {
      if (i < ghosts || i >= ni - ghosts || j < ghosts || j >= nj - ghosts) continue;
    }

    int stride, npoints_dir, idx_dir;
    if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
    else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
    else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

    int qC = tid;
    int qL = tid - stride;
    int qR = tid + stride;

    for (int v = 0; v < nvars; v++) {
      double deriv;
      if (idx_dir == 0) {
        deriv = f[qR*nvars+v] - f[qC*nvars+v];
      } else if (idx_dir == npoints_dir - 1) {
        deriv = f[qC*nvars+v] - f[qL*nvars+v];
      } else if (bias > 0) {
        deriv = f[qR*nvars+v] - f[qC*nvars+v];
      } else if (bias < 0) {
        deriv = f[qC*nvars+v] - f[qL*nvars+v];
      } else {
        deriv = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
      }
      Df[qC*nvars+v] = deriv;
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int total_points = ni * nj * nk;
  int gridSize = (total_points + blockSize - 1) / blockSize;

  /* Use generic kernel - specialized kernels need ghosts parameter update */
  GPU_KERNEL_LAUNCH(gpu_first_derivative_first_order_3d_kernel, gridSize, blockSize)(
    Df, f, nvars, ni, nj, nk, ghosts, dir, bias
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_first_derivative_second_order_3d(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - only process interior (i,k) positions to match per-line behavior */
  int total = ni * nj * nk;
  for (int tid = 0; tid < total; tid++) {
    int i = tid % ni;
    int j = (tid / ni) % nj;
    int k = tid / (ni * nj);

    /* Skip ghost regions in non-derivative directions */
    if (dir == 0) {
      if (j < ghosts || j >= nj - ghosts || k < ghosts || k >= nk - ghosts) continue;
    } else if (dir == 1) {
      if (i < ghosts || i >= ni - ghosts || k < ghosts || k >= nk - ghosts) continue;
    } else {
      if (i < ghosts || i >= ni - ghosts || j < ghosts || j >= nj - ghosts) continue;
    }

    int stride, npoints_dir, idx_dir;
    if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
    else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
    else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

    int qC = tid;
    int qL = tid - stride;
    int qR = tid + stride;

    for (int v = 0; v < nvars; v++) {
      double deriv;
      if (idx_dir == 0) {
        int qRR = tid + 2 * stride;
        deriv = 0.5 * (-3.0*f[qC*nvars+v] + 4.0*f[qR*nvars+v] - f[qRR*nvars+v]);
      } else if (idx_dir == npoints_dir - 1) {
        int qLL = tid - 2 * stride;
        deriv = 0.5 * (3.0*f[qC*nvars+v] - 4.0*f[qL*nvars+v] + f[qLL*nvars+v]);
      } else {
        deriv = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
      }
      Df[qC*nvars+v] = deriv;
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int total_points = ni * nj * nk;
  int gridSize = (total_points + blockSize - 1) / blockSize;

  /* Use generic kernel - specialized kernels need ghosts parameter update */
  GPU_KERNEL_LAUNCH(gpu_first_derivative_second_order_3d_kernel, gridSize, blockSize)(
    Df, f, nvars, ni, nj, nk, ghosts, dir
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_first_derivative_fourth_order_3d(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - only process interior (i,k) positions to match per-line behavior */
  const double one_twelve = 1.0/12.0;
  int total = ni * nj * nk;
  for (int tid = 0; tid < total; tid++) {
    int i = tid % ni;
    int j = (tid / ni) % nj;
    int k = tid / (ni * nj);

    /* Skip ghost regions in non-derivative directions */
    if (dir == 0) {
      if (j < ghosts || j >= nj - ghosts || k < ghosts || k >= nk - ghosts) continue;
    } else if (dir == 1) {
      if (i < ghosts || i >= ni - ghosts || k < ghosts || k >= nk - ghosts) continue;
    } else {
      if (i < ghosts || i >= ni - ghosts || j < ghosts || j >= nj - ghosts) continue;
    }

    int stride, npoints_dir, idx_dir;
    if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
    else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
    else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

    int qC = tid;

    for (int v = 0; v < nvars; v++) {
      double deriv;
      if (idx_dir == 0) {
        int qp1 = tid + stride;
        int qp2 = tid + 2*stride;
        int qp3 = tid + 3*stride;
        int qp4 = tid + 4*stride;
        deriv = one_twelve * (-25.0*f[qC*nvars+v] + 48.0*f[qp1*nvars+v] - 36.0*f[qp2*nvars+v] + 16.0*f[qp3*nvars+v] - 3.0*f[qp4*nvars+v]);
      } else if (idx_dir == 1) {
        int qm1 = tid - stride;
        int qp1 = tid + stride;
        int qp2 = tid + 2*stride;
        int qp3 = tid + 3*stride;
        deriv = one_twelve * (-3.0*f[qm1*nvars+v] - 10.0*f[qC*nvars+v] + 18.0*f[qp1*nvars+v] - 6.0*f[qp2*nvars+v] + f[qp3*nvars+v]);
      } else if (idx_dir >= 2 && idx_dir < npoints_dir - 2) {
        int qL = tid - stride;
        int qR = tid + stride;
        int qLL = tid - 2*stride;
        int qRR = tid + 2*stride;
        deriv = one_twelve * (f[qLL*nvars+v] - 8.0*f[qL*nvars+v] + 8.0*f[qR*nvars+v] - f[qRR*nvars+v]);
      } else if (idx_dir == npoints_dir - 2) {
        int qm3 = tid - 3*stride;
        int qm2 = tid - 2*stride;
        int qm1 = tid - stride;
        int qp1 = tid + stride;
        deriv = one_twelve * (-f[qm3*nvars+v] + 6.0*f[qm2*nvars+v] - 18.0*f[qm1*nvars+v] + 10.0*f[qC*nvars+v] + 3.0*f[qp1*nvars+v]);
      } else {
        int qm4 = tid - 4*stride;
        int qm3 = tid - 3*stride;
        int qm2 = tid - 2*stride;
        int qm1 = tid - stride;
        deriv = one_twelve * (3.0*f[qm4*nvars+v] - 16.0*f[qm3*nvars+v] + 36.0*f[qm2*nvars+v] - 48.0*f[qm1*nvars+v] + 25.0*f[qC*nvars+v]);
      }
      Df[qC*nvars+v] = deriv;
    }
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int total_points = ni * nj * nk;
  int gridSize = (total_points + blockSize - 1) / blockSize;

  /* Use generic kernel - specialized kernels need ghosts parameter update */
  GPU_KERNEL_LAUNCH(gpu_first_derivative_fourth_order_3d_kernel, gridSize, blockSize)(
    Df, f, nvars, ni, nj, nk, ghosts, dir
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

} /* extern "C" */

