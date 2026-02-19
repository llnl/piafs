/*! @file gpu_parabolic_launch.c
    @brief GPU parabolic kernel launch wrappers
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_parabolic.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_ns3d_viscous_flux_x(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double mu, double kappa, double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified version */
  static const double two_third = 2.0/3.0;
  for (int i = 0; i < npoints; i++) {
    double uvel = Q[i*nvars + 1];
    double vvel = Q[i*nvars + 2];
    double wvel = Q[i*nvars + 3];
    double ux = QDerivX[i*nvars + 1];
    double vx = QDerivX[i*nvars + 2];
    double wx = QDerivX[i*nvars + 3];
    double uy = QDerivY[i*nvars + 1];
    double vy = QDerivY[i*nvars + 2];
    double uz = QDerivZ[i*nvars + 1];
    double wz = QDerivZ[i*nvars + 3];
    double Tx = QDerivX[i*nvars + 4];

    double tau_xx = two_third * (mu * inv_Re) * (2*ux - vy - wz);
    double tau_xy = (mu * inv_Re) * (uy + vx);
    double tau_xz = (mu * inv_Re) * (uz + wx);
    double qx = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Tx;

    FViscous[i*nvars + 0] = 0.0;
    FViscous[i*nvars + 1] = tau_xx;
    FViscous[i*nvars + 2] = tau_xy;
    FViscous[i*nvars + 3] = tau_xz;
    FViscous[i*nvars + 4] = uvel*tau_xx + vvel*tau_xy + wvel*tau_xz + qx;
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_viscous_flux_x_kernel, gridSize, blockSize)(
    FViscous, Q, QDerivX, QDerivY, QDerivZ, nvars, npoints, mu, kappa, inv_Re, inv_gamma_m1, inv_Pr
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_ns3d_viscous_flux_y(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double mu, double kappa, double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  static const double two_third = 2.0/3.0;
  for (int i = 0; i < npoints; i++) {
    double uvel = Q[i*nvars + 1];
    double vvel = Q[i*nvars + 2];
    double wvel = Q[i*nvars + 3];
    double ux = QDerivX[i*nvars + 1];
    double vx = QDerivX[i*nvars + 2];
    double uy = QDerivY[i*nvars + 1];
    double vy = QDerivY[i*nvars + 2];
    double wy = QDerivY[i*nvars + 3];
    double vz = QDerivZ[i*nvars + 2];
    double wz = QDerivZ[i*nvars + 3];
    double Ty = QDerivY[i*nvars + 4];

    double tau_yx = (mu * inv_Re) * (uy + vx);
    double tau_yy = two_third * (mu * inv_Re) * (-ux + 2*vy - wz);
    double tau_yz = (mu * inv_Re) * (vz + wy);
    double qy = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Ty;

    FViscous[i*nvars + 0] = 0.0;
    FViscous[i*nvars + 1] = tau_yx;
    FViscous[i*nvars + 2] = tau_yy;
    FViscous[i*nvars + 3] = tau_yz;
    FViscous[i*nvars + 4] = uvel*tau_yx + vvel*tau_yy + wvel*tau_yz + qy;
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_viscous_flux_y_kernel, gridSize, blockSize)(
    FViscous, Q, QDerivX, QDerivY, QDerivZ, nvars, npoints, mu, kappa, inv_Re, inv_gamma_m1, inv_Pr
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_ns3d_viscous_flux_z(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double mu, double kappa, double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  static const double two_third = 2.0/3.0;
  for (int i = 0; i < npoints; i++) {
    double uvel = Q[i*nvars + 1];
    double vvel = Q[i*nvars + 2];
    double wvel = Q[i*nvars + 3];
    double ux = QDerivX[i*nvars + 1];
    double wx = QDerivX[i*nvars + 3];
    double vy = QDerivY[i*nvars + 2];
    double wy = QDerivY[i*nvars + 3];
    double uz = QDerivZ[i*nvars + 1];
    double vz = QDerivZ[i*nvars + 2];
    double wz = QDerivZ[i*nvars + 3];
    double Tz = QDerivZ[i*nvars + 4];

    double tau_zx = (mu * inv_Re) * (uz + wx);
    double tau_zy = (mu * inv_Re) * (vz + wy);
    double tau_zz = two_third * (mu * inv_Re) * (-ux - vy + 2*wz);
    double qz = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Tz;

    FViscous[i*nvars + 0] = 0.0;
    FViscous[i*nvars + 1] = tau_zx;
    FViscous[i*nvars + 2] = tau_zy;
    FViscous[i*nvars + 3] = tau_zz;
    FViscous[i*nvars + 4] = uvel*tau_zx + vvel*tau_zy + wvel*tau_zz + qz;
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_viscous_flux_z_kernel, gridSize, blockSize)(
    FViscous, Q, QDerivX, QDerivY, QDerivZ, nvars, npoints, mu, kappa, inv_Re, inv_gamma_m1, inv_Pr
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_ns3d_get_primitive(
  double *Q, const double *u, int nvars, int npoints, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  for (int i = 0; i < npoints; i++) {
    double rho = u[i*nvars + 0];
    double rho_u = u[i*nvars + 1];
    double rho_v = u[i*nvars + 2];
    double rho_w = u[i*nvars + 3];
    double e = u[i*nvars + 4];

    double uvel = (rho == 0) ? 0.0 : rho_u / rho;
    double vvel = (rho == 0) ? 0.0 : rho_v / rho;
    double wvel = (rho == 0) ? 0.0 : rho_w / rho;
    double vsq = uvel*uvel + vvel*vvel + wvel*wvel;
    double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    double T = gamma * P / rho;  /* matches CPU: Q[p+4] = physics->gamma*pressure/Q[p+0] */

    Q[i*nvars + 0] = rho;
    Q[i*nvars + 1] = uvel;
    Q[i*nvars + 2] = vvel;
    Q[i*nvars + 3] = wvel;
    Q[i*nvars + 4] = T;
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_ns3d_get_primitive_kernel, gridSize, blockSize)(
    Q, u, nvars, npoints, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_scale_array_with_dxinv(
  double *x, const double *dxinv, int nvars, int npoints,
  int ndims, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int dir_offset, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified */
  for (int i = 0; i < npoints; i++) {
    /* Would need full index computation */
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_scale_array_with_dxinv_kernel, gridSize, blockSize)(
    x, dxinv, nvars, npoints, ndims, dim, stride_with_ghosts, ghosts, dir, dir_offset
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_add_scaled_derivative(
  double *par, const double *FDeriv, const double *dxinv,
  int nvars, int npoints, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, int dir_offset, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified */
  for (int i = 0; i < npoints; i++) {
    /* Would need full index computation */
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_add_scaled_derivative_kernel, gridSize, blockSize)(
    par, FDeriv, dxinv, nvars, npoints, ndims, dim, stride_with_ghosts, ghosts, dir, dir_offset
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

