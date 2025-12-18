/*! @file gpu_parabolic_launch.cu
    @brief GPU parabolic kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_parabolic.h>

#define DEFAULT_BLOCK_SIZE 256

extern "C" {
void gpu_launch_ns2d_get_primitive(double *Q, const double *u, int nvars, int npoints, double gamma, int blockSize)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns2d_get_primitive_kernel, gridSize, blockSize)(Q, u, nvars, npoints, gamma);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_ns2d_viscous_flux_x(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns2d_viscous_flux_x_kernel, gridSize, blockSize)(
    FViscous, Q, QDerivX, QDerivY, nvars, npoints, Tref, T0, TS, TA, TB, inv_Re, inv_gamma_m1, inv_Pr
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_ns2d_viscous_flux_y(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns2d_viscous_flux_y_kernel, gridSize, blockSize)(
    FViscous, Q, QDerivX, QDerivY, nvars, npoints, Tref, T0, TS, TA, TB, inv_Re, inv_gamma_m1, inv_Pr
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}
} /* extern "C" */

