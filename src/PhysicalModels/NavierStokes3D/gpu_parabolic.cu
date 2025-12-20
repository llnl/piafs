/*! @file gpu_parabolic.cu
    @brief GPU kernels for parabolic (viscous) term computation
*/

#include <gpu.h>
#include <math.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Element-wise scale array with dxinv per grid point
   Scales each grid point's nvars values by the corresponding dxinv value
*/
GPU_KERNEL void gpu_scale_array_with_dxinv_kernel(
  double *x,                    /* input/output: array to scale */
  const double *dxinv,          /* input: scale factors (1/dx) */
  int nvars,                    /* number of variables per grid point */
  int npoints,                  /* number of grid points */
  int ndims,                    /* number of dimensions */
  const int *dim,               /* dimensions (without ghosts) */
  const int *stride_with_ghosts, /* stride array */
  int ghosts,                   /* number of ghost points */
  int dir,                      /* direction for dxinv */
  int dir_offset                /* offset in dxinv array */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    /* Reconstruct multi-dimensional index */
    int index[3];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      index[i] = temp % (dim[i] + 2 * ghosts);
      temp /= (dim[i] + 2 * ghosts);
    }
    
    /* Get dxinv value for this point (index[dir] is 0-based including ghosts) */
    double scale = dxinv[dir_offset + index[dir]];
    
    /* Scale all variables at this point */
    int p = 0;
    for (int i = 0; i < ndims; i++) {
      p += index[i] * stride_with_ghosts[i];
    }
    
    for (int v = 0; v < nvars; v++) {
      x[p * nvars + v] *= scale;
    }
  }
}

/* Kernel: Add scaled derivative to parabolic term
   par[p] += dxinv * FDeriv[p] for each grid point
*/
GPU_KERNEL void gpu_add_scaled_derivative_kernel(
  double *par,                  /* output: parabolic term */
  const double *FDeriv,         /* input: derivative to add */
  const double *dxinv,          /* input: scale factors */
  int nvars,                    /* number of variables */
  int npoints,                  /* number of grid points (without ghosts) */
  int ndims,                    /* number of dimensions */
  const int *dim,               /* dimensions (without ghosts) */
  const int *stride_with_ghosts, /* stride array */
  int ghosts,                   /* number of ghost points */
  int dir,                      /* direction for dxinv */
  int dir_offset                /* offset in dxinv array */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    /* Reconstruct multi-dimensional index (without ghosts) */
    int index[3];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      index[i] = temp % dim[i];
      temp /= dim[i];
    }
    
    /* Get dxinv value for this point */
    double scale = dxinv[dir_offset + ghosts + index[dir]];
    
    /* Compute 1D array index (with ghosts) */
    int p = 0;
    for (int i = 0; i < ndims; i++) {
      p += (index[i] + ghosts) * stride_with_ghosts[i];
    }
    
    /* Add scaled derivative for all variables */
    for (int v = 0; v < nvars; v++) {
      par[p * nvars + v] += scale * FDeriv[p * nvars + v];
    }
  }
}

/* Kernel: Compute viscous flux in X direction for 3D Navier-Stokes */
GPU_KERNEL void gpu_ns3d_viscous_flux_x_kernel(
  double *FViscous,     /* output: viscous flux */
  const double *Q,       /* input: primitive variables (rho, u, v, w, T) */
  const double *QDerivX, /* input: derivatives in x */
  const double *QDerivY, /* input: derivatives in y */
  const double *QDerivZ, /* input: derivatives in z */
  int nvars,             /* number of variables */
  int npoints,           /* number of grid points */
  double Tref,           /* reference temperature */
  double T0,             /* T_0 (viscosity/conductivity coeff) */
  double TS,             /* T_S (viscosity/conductivity coeff) */
  double TA,             /* T_A (conductivity coeff) */
  double TB,             /* T_B (conductivity coeff) */
  double inv_Re,         /* 1/Reynolds number */
  double inv_gamma_m1,   /* 1/(gamma-1) */
  double inv_Pr          /* 1/Prandtl number */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double uvel = Q[idx*nvars + 1];
    double vvel = Q[idx*nvars + 2];
    double wvel = Q[idx*nvars + 3];
    double T = Q[idx*nvars + 4];
    
    double ux = QDerivX[idx*nvars + 1];
    double vx = QDerivX[idx*nvars + 2];
    double wx = QDerivX[idx*nvars + 3];
    double uy = QDerivY[idx*nvars + 1];
    double vy = QDerivY[idx*nvars + 2];
    double uz = QDerivZ[idx*nvars + 1];
    double wz = QDerivZ[idx*nvars + 3];
    double Tx = QDerivX[idx*nvars + 4];
    
    /* Compute viscosity and conductivity coefficients from temperature */
    double T_d = T * Tref;
    double mu = exp(1.5 * log(T_d / T0)) * (T0 + TS) / (T_d + TS);
    double kappa = exp(1.5 * log(T_d / T0)) 
                   * (T0 + TA * exp(-TB / T0)) 
                   / (T_d + TA * exp(-TB / T_d));
    
    const double two_third = 2.0/3.0;
    
    /* Compute stress tensor components */
    double tau_xx = two_third * (mu * inv_Re) * (2*ux - vy - wz);
    double tau_xy = (mu * inv_Re) * (uy + vx);
    double tau_xz = (mu * inv_Re) * (uz + wx);
    double qx = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Tx;
    
    /* Viscous flux */
    FViscous[idx*nvars + 0] = 0.0;
    FViscous[idx*nvars + 1] = tau_xx;
    FViscous[idx*nvars + 2] = tau_xy;
    FViscous[idx*nvars + 3] = tau_xz;
    FViscous[idx*nvars + 4] = uvel*tau_xx + vvel*tau_xy + wvel*tau_xz + qx;
  }
}

/* Kernel: Compute viscous flux in Y direction for 3D Navier-Stokes */
GPU_KERNEL void gpu_ns3d_viscous_flux_y_kernel(
  double *FViscous,     /* output: viscous flux */
  const double *Q,       /* input: primitive variables */
  const double *QDerivX, /* input: derivatives in x */
  const double *QDerivY, /* input: derivatives in y */
  const double *QDerivZ, /* input: derivatives in z */
  int nvars,
  int npoints,
  double Tref,           /* reference temperature */
  double T0,             /* T_0 (viscosity/conductivity coeff) */
  double TS,             /* T_S (viscosity/conductivity coeff) */
  double TA,             /* T_A (conductivity coeff) */
  double TB,             /* T_B (conductivity coeff) */
  double inv_Re,
  double inv_gamma_m1,
  double inv_Pr
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double uvel = Q[idx*nvars + 1];
    double vvel = Q[idx*nvars + 2];
    double wvel = Q[idx*nvars + 3];
    double T = Q[idx*nvars + 4];
    
    double ux = QDerivX[idx*nvars + 1];
    double vx = QDerivX[idx*nvars + 2];
    double uy = QDerivY[idx*nvars + 1];
    double vy = QDerivY[idx*nvars + 2];
    double wy = QDerivY[idx*nvars + 3];
    double vz = QDerivZ[idx*nvars + 2];
    double wz = QDerivZ[idx*nvars + 3];
    double Ty = QDerivY[idx*nvars + 4];
    
    /* Compute viscosity and conductivity coefficients from temperature */
    double T_d = T * Tref;
    double mu = exp(1.5 * log(T_d / T0)) * (T0 + TS) / (T_d + TS);
    double kappa = exp(1.5 * log(T_d / T0)) 
                   * (T0 + TA * exp(-TB / T0)) 
                   / (T_d + TA * exp(-TB / T_d));
    
    const double two_third = 2.0/3.0;
    
    double tau_yx = (mu * inv_Re) * (uy + vx);
    double tau_yy = two_third * (mu * inv_Re) * (-ux + 2*vy - wz);
    double tau_yz = (mu * inv_Re) * (vz + wy);
    double qy = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Ty;
    
    FViscous[idx*nvars + 0] = 0.0;
    FViscous[idx*nvars + 1] = tau_yx;
    FViscous[idx*nvars + 2] = tau_yy;
    FViscous[idx*nvars + 3] = tau_yz;
    FViscous[idx*nvars + 4] = uvel*tau_yx + vvel*tau_yy + wvel*tau_yz + qy;
  }
}

/* Kernel: Compute viscous flux in Z direction for 3D Navier-Stokes */
GPU_KERNEL void gpu_ns3d_viscous_flux_z_kernel(
  double *FViscous,     /* output: viscous flux */
  const double *Q,       /* input: primitive variables */
  const double *QDerivX, /* input: derivatives in x */
  const double *QDerivY, /* input: derivatives in y */
  const double *QDerivZ, /* input: derivatives in z */
  int nvars,
  int npoints,
  double Tref,           /* reference temperature */
  double T0,             /* T_0 (viscosity/conductivity coeff) */
  double TS,             /* T_S (viscosity/conductivity coeff) */
  double TA,             /* T_A (conductivity coeff) */
  double TB,             /* T_B (conductivity coeff) */
  double inv_Re,
  double inv_gamma_m1,
  double inv_Pr
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double uvel = Q[idx*nvars + 1];
    double vvel = Q[idx*nvars + 2];
    double wvel = Q[idx*nvars + 3];
    double T = Q[idx*nvars + 4];
    
    double ux = QDerivX[idx*nvars + 1];
    double wx = QDerivX[idx*nvars + 3];
    double vy = QDerivY[idx*nvars + 2];
    double wy = QDerivY[idx*nvars + 3];
    double uz = QDerivZ[idx*nvars + 1];
    double vz = QDerivZ[idx*nvars + 2];
    double wz = QDerivZ[idx*nvars + 3];
    double Tz = QDerivZ[idx*nvars + 4];
    
    /* Compute viscosity and conductivity coefficients from temperature */
    double T_d = T * Tref;
    double mu = exp(1.5 * log(T_d / T0)) * (T0 + TS) / (T_d + TS);
    double kappa = exp(1.5 * log(T_d / T0)) 
                   * (T0 + TA * exp(-TB / T0)) 
                   / (T_d + TA * exp(-TB / T_d));
    
    const double two_third = 2.0/3.0;
    
    double tau_zx = (mu * inv_Re) * (uz + wx);
    double tau_zy = (mu * inv_Re) * (vz + wy);
    double tau_zz = two_third * (mu * inv_Re) * (-ux - vy + 2*wz);
    double qz = (kappa * inv_Re * inv_gamma_m1 * inv_Pr) * Tz;
    
    FViscous[idx*nvars + 0] = 0.0;
    FViscous[idx*nvars + 1] = tau_zx;
    FViscous[idx*nvars + 2] = tau_zy;
    FViscous[idx*nvars + 3] = tau_zz;
    FViscous[idx*nvars + 4] = uvel*tau_zx + vvel*tau_zy + wvel*tau_zz + qz;
  }
}

/* Kernel: Compute primitive variables from conserved variables */
GPU_KERNEL void gpu_ns3d_get_primitive_kernel(
  double *Q,            /* output: primitive variables (rho, u, v, w, T) */
  const double *u,      /* input: conserved variables (rho, rho*u, rho*v, rho*w, e) */
  int nvars,
  int npoints,
  double gamma
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < npoints) {
    double rho = u[idx*nvars + 0];
    double rho_u = u[idx*nvars + 1];
    double rho_v = u[idx*nvars + 2];
    double rho_w = u[idx*nvars + 3];
    double e = u[idx*nvars + 4];
    
    double uvel = (rho == 0) ? 0.0 : rho_u / rho;
    double vvel = (rho == 0) ? 0.0 : rho_v / rho;
    double wvel = (rho == 0) ? 0.0 : rho_w / rho;
    double vsq = uvel*uvel + vvel*vvel + wvel*wvel;
    double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    double T = gamma * P / rho;  /* matches CPU: Q[p+4] = physics->gamma*pressure/Q[p+0] */
    
    Q[idx*nvars + 0] = rho;
    Q[idx*nvars + 1] = uvel;
    Q[idx*nvars + 2] = vvel;
    Q[idx*nvars + 3] = wvel;
    Q[idx*nvars + 4] = T;
  }
}

