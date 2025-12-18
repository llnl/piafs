/*! @file gpu_parabolic.cu
    @brief GPU kernels for NavierStokes2D parabolic (viscous) term computation
*/

#include <gpu.h>
#include <math.h>
#include <physicalmodels/navierstokes2d.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Compute primitive variables from conserved variables for NS2D */
GPU_KERNEL void gpu_ns2d_get_primitive_kernel(
  double *Q, const double *u, int nvars, int npoints, double gamma
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double rho = u[idx*nvars + 0];
    double rho_u = u[idx*nvars + 1];
    double rho_v = u[idx*nvars + 2];
    double e = u[idx*nvars + 3];
    double uvel = (rho == 0) ? 0.0 : rho_u / rho;
    double vvel = (rho == 0) ? 0.0 : rho_v / rho;
    double vsq = uvel*uvel + vvel*vvel;
    double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    double T = gamma * P / rho;
    Q[idx*nvars + 0] = rho;
    Q[idx*nvars + 1] = uvel;
    Q[idx*nvars + 2] = vvel;
    Q[idx*nvars + 3] = T;
    for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
      Q[idx*nvars + m_i] = u[idx*nvars + m_i];
    }
  }
}

/* Kernel: Compute viscous flux in X direction for 2D Navier-Stokes */
GPU_KERNEL void gpu_ns2d_viscous_flux_x_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double uvel = Q[idx*nvars + 1];
    double vvel = Q[idx*nvars + 2];
    double T = Q[idx*nvars + 3];
    double ux = QDerivX[idx*nvars + 1];
    double vx = QDerivX[idx*nvars + 2];
    double uy = QDerivY[idx*nvars + 1];
    double vy = QDerivY[idx*nvars + 2];
    double Tx = QDerivX[idx*nvars + 3];
    double T_d = T * Tref;
    double mu = exp(1.5 * log(T_d / T0)) * (T0 + TS) / (T_d + TS);
    double kappa = exp(1.5 * log(T_d / T0)) * (T0 + TA * exp(-TB / T0)) / (T_d + TA * exp(-TB / T_d));
    static const double two_third = 2.0/3.0;
    double tau_xx = two_third * (mu * inv_Re) * (2*ux - vy);
    double tau_xy = (mu * inv_Re) * (uy + vx);
    double qx = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Tx;
    FViscous[idx*nvars + 0] = 0.0;
    FViscous[idx*nvars + 1] = tau_xx;
    FViscous[idx*nvars + 2] = tau_xy;
    FViscous[idx*nvars + 3] = uvel*tau_xx + vvel*tau_xy + qx;
  }
}

/* Kernel: Compute viscous flux in Y direction for 2D Navier-Stokes */
GPU_KERNEL void gpu_ns2d_viscous_flux_y_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double uvel = Q[idx*nvars + 1];
    double vvel = Q[idx*nvars + 2];
    double T = Q[idx*nvars + 3];
    double ux = QDerivX[idx*nvars + 1];
    double vx = QDerivX[idx*nvars + 2];
    double uy = QDerivY[idx*nvars + 1];
    double vy = QDerivY[idx*nvars + 2];
    double Ty = QDerivY[idx*nvars + 3];
    double T_d = T * Tref;
    double mu = exp(1.5 * log(T_d / T0)) * (T0 + TS) / (T_d + TS);
    double kappa = exp(1.5 * log(T_d / T0)) * (T0 + TA * exp(-TB / T0)) / (T_d + TA * exp(-TB / T_d));
    static const double two_third = 2.0/3.0;
    double tau_yx = (mu * inv_Re) * (uy + vx);
    double tau_yy = two_third * (mu * inv_Re) * (-ux + 2*vy);
    double qy = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Ty;
    FViscous[idx*nvars + 0] = 0.0;
    FViscous[idx*nvars + 1] = tau_yx;
    FViscous[idx*nvars + 2] = tau_yy;
    FViscous[idx*nvars + 3] = uvel*tau_yx + vvel*tau_yy + qy;
  }
}

