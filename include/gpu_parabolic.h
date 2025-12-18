/*! @file gpu_parabolic.h
    @brief GPU parabolic kernel declarations
*/

#ifndef _GPU_PARABOLIC_H_
#define _GPU_PARABOLIC_H_

#include <gpu.h>

/* Kernel annotations must only be visible to CUDA/HIP compilers. */
#if defined(GPU_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP) && (defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__))
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* NavierStokes3D kernels */
GPU_KERNEL void gpu_scale_array_with_dxinv_kernel(
  double *x, const double *dxinv, int nvars, int npoints, int ndims,
  const int *dim, const int *stride_with_ghosts, int ghosts, int dir, int dir_offset
);

GPU_KERNEL void gpu_add_scaled_derivative_kernel(
  double *x, const double *y, const double *dxinv, int nvars, int npoints, int ndims,
  const int *dim, const int *stride_with_ghosts, int ghosts, int dir, int dir_offset
);

GPU_KERNEL void gpu_ns3d_get_primitive_kernel(double *Q, const double *u, int nvars, int npoints, double gamma);
GPU_KERNEL void gpu_ns3d_viscous_flux_x_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
);
GPU_KERNEL void gpu_ns3d_viscous_flux_y_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
);
GPU_KERNEL void gpu_ns3d_viscous_flux_z_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
);

/* NavierStokes2D kernels */
GPU_KERNEL void gpu_ns2d_get_primitive_kernel(double *Q, const double *u, int nvars, int npoints, double gamma);
GPU_KERNEL void gpu_ns2d_viscous_flux_x_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
);
GPU_KERNEL void gpu_ns2d_viscous_flux_y_kernel(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr
);

/* Launch wrappers */
#ifdef __cplusplus
extern "C" {
#endif

void gpu_launch_scale_array_with_dxinv(
  double *x, const double *dxinv, int nvars, int npoints, int ndims,
  const int *dim, const int *stride_with_ghosts, int ghosts, int dir, int dir_offset, int blockSize
);
void gpu_launch_add_scaled_derivative(
  double *x, const double *y, const double *dxinv, int nvars, int npoints, int ndims,
  const int *dim, const int *stride_with_ghosts, int ghosts, int dir, int dir_offset, int blockSize
);
void gpu_launch_ns3d_get_primitive(double *Q, const double *u, int nvars, int npoints, double gamma, int blockSize);
void gpu_launch_ns3d_viscous_flux_x(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
);
void gpu_launch_ns3d_viscous_flux_y(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
);
void gpu_launch_ns3d_viscous_flux_z(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY, const double *QDerivZ,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
);
void gpu_launch_ns2d_get_primitive(double *Q, const double *u, int nvars, int npoints, double gamma, int blockSize);
void gpu_launch_ns2d_viscous_flux_x(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
);
void gpu_launch_ns2d_viscous_flux_y(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_PARABOLIC_H_ */
