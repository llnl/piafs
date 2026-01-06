/*! @file gpu_first_derivative.c
    @brief GPU-enabled first derivative functions
*/

#include <gpu.h>
#include <gpu_derivative.h>
#include <gpu_runtime.h>
#include <hypar.h>
#include <mpivars.h>
#include <arrayfunctions.h>

/* GPU-enabled second order central first derivative */
int GPUFirstDerivativeSecondOrderCentral(
  double *Df,
  double *f,
  int dir,
  int bias,
  void *s,
  void *m
)
{
  HyPar *solver = (HyPar*) s;
  int ghosts = solver->ghosts;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int *dim = solver->dim_local;

  if ((!Df) || (!f)) {
    fprintf(stderr, "Error in GPUFirstDerivativeSecondOrderCentral(): input arrays not allocated.\n");
    return 1;
  }

  if (GPUShouldUse()) {
    /* Use optimized 3D kernel for 3D problems - single kernel launch */
    if (ndims == 3) {
      int ni = dim[0] + 2 * ghosts;
      int nj = dim[1] + 2 * ghosts;
      int nk = dim[2] + 2 * ghosts;

      gpu_launch_first_derivative_second_order_3d(
        Df, f, nvars, ni, nj, nk, ghosts, dir, 256
      );

      if (GPUShouldSyncEveryOp()) GPUSync();
      return 0;
    }

    /* Fallback to per-line approach for 1D/2D */
    int bounds_outer[ndims];
    _ArrayCopy1D_(dim, bounds_outer, ndims);
    bounds_outer[dir] = 1;
    int N_outer;
    _ArrayProduct1D_(bounds_outer, ndims, N_outer);

    int stride = solver->stride_with_ghosts[dir];
    int npoints_line = dim[dir] + 2 * ghosts;

    for (int line = 0; line < N_outer; line++) {
      int index_outer[ndims];
      _ArrayIndexnD_(ndims, line, bounds_outer, index_outer, 0);

      int base_offset = 0;
      for (int d = 0; d < ndims; d++) {
        if (d != dir) {
          base_offset += (index_outer[d] + ghosts) * solver->stride_with_ghosts[d];
        }
      }

      double *f_line = f + base_offset * nvars;
      double *Df_line = Df + base_offset * nvars;

      gpu_launch_first_derivative_second_order(
        Df_line, f_line, nvars, npoints_line, ghosts, stride, 256
      );
    }

    if (GPUShouldSyncEveryOp()) GPUSync();
    return 0;
  } else {
    return 1;
  }
}

/* GPU-enabled fourth order central first derivative */
int GPUFirstDerivativeFourthOrderCentral(
  double *Df,
  double *f,
  int dir,
  int bias,
  void *s,
  void *m
)
{
  HyPar *solver = (HyPar*) s;
  int ghosts = solver->ghosts;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int *dim = solver->dim_local;

  if ((!Df) || (!f)) {
    fprintf(stderr, "Error in GPUFirstDerivativeFourthOrderCentral(): input arrays not allocated.\n");
    return 1;
  }

  if (GPUShouldUse()) {
    /* Use optimized 3D kernel for 3D problems - single kernel launch */
    if (ndims == 3) {
      int ni = dim[0] + 2 * ghosts;
      int nj = dim[1] + 2 * ghosts;
      int nk = dim[2] + 2 * ghosts;

      gpu_launch_first_derivative_fourth_order_3d(
        Df, f, nvars, ni, nj, nk, ghosts, dir, 256
      );

      if (GPUShouldSyncEveryOp()) GPUSync();
      return 0;
    }

    /* Fallback to per-line approach for 1D/2D */
    int bounds_outer[ndims];
    _ArrayCopy1D_(dim, bounds_outer, ndims);
    bounds_outer[dir] = 1;
    int N_outer;
    _ArrayProduct1D_(bounds_outer, ndims, N_outer);

    int stride = solver->stride_with_ghosts[dir];
    int npoints_line = dim[dir] + 2 * ghosts;

    for (int line = 0; line < N_outer; line++) {
      int index_outer[ndims];
      _ArrayIndexnD_(ndims, line, bounds_outer, index_outer, 0);

      int base_offset = 0;
      for (int d = 0; d < ndims; d++) {
        if (d != dir) {
          base_offset += (index_outer[d] + ghosts) * solver->stride_with_ghosts[d];
        }
      }

      double *f_line = f + base_offset * nvars;
      double *Df_line = Df + base_offset * nvars;

      gpu_launch_first_derivative_fourth_order(
        Df_line, f_line, nvars, npoints_line, ghosts, stride, 256
      );
    }

    if (GPUShouldSyncEveryOp()) GPUSync();
    return 0;
  } else {
    return 1;
  }
}

/* GPU-enabled first order first derivative */
int GPUFirstDerivativeFirstOrder(
  double *Df,
  double *f,
  int dir,
  int bias,
  void *s,
  void *m
)
{
  HyPar *solver = (HyPar*) s;
  int ghosts = solver->ghosts;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int *dim = solver->dim_local;

  if ((!Df) || (!f)) {
    fprintf(stderr, "Error in GPUFirstDerivativeFirstOrder(): input arrays not allocated.\n");
    return 1;
  }

  if (GPUShouldUse()) {
    double bias_double = (bias > 0) ? 1.0 : ((bias < 0) ? -1.0 : 0.0);

    /* Use optimized 3D kernel for 3D problems - single kernel launch */
    if (ndims == 3) {
      int ni = dim[0] + 2 * ghosts;
      int nj = dim[1] + 2 * ghosts;
      int nk = dim[2] + 2 * ghosts;

      gpu_launch_first_derivative_first_order_3d(
        Df, f, nvars, ni, nj, nk, ghosts, dir, bias_double, 256
      );

      if (GPUShouldSyncEveryOp()) GPUSync();
      return 0;
    }

    /* Fallback to per-line approach for 1D/2D */
    int bounds_outer[ndims];
    _ArrayCopy1D_(dim, bounds_outer, ndims);
    bounds_outer[dir] = 1;
    int N_outer;
    _ArrayProduct1D_(bounds_outer, ndims, N_outer);

    int stride = solver->stride_with_ghosts[dir];
    int npoints_line = dim[dir] + 2 * ghosts;

    for (int line = 0; line < N_outer; line++) {
      int index_outer[ndims];
      _ArrayIndexnD_(ndims, line, bounds_outer, index_outer, 0);

      int base_offset = 0;
      for (int d = 0; d < ndims; d++) {
        if (d != dir) {
          base_offset += (index_outer[d] + ghosts) * solver->stride_with_ghosts[d];
        }
      }

      double *f_line = f + base_offset * nvars;
      double *Df_line = Df + base_offset * nvars;

      gpu_launch_first_derivative_first_order(
        Df_line, f_line, nvars, npoints_line, ghosts, stride, bias_double, 256
      );
    }

    if (GPUShouldSyncEveryOp()) GPUSync();
    return 0;
  } else {
    return 1;
  }
}

