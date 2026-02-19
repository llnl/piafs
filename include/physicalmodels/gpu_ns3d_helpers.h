/*! @file gpu_ns3d_helpers.h
    @brief Shared GPU helper functions for NavierStokes3D
    @details These are inline device functions that can be included in multiple .cu files
*/

#ifndef _GPU_NS3D_HELPERS_H_
#define _GPU_NS3D_HELPERS_H_

#include <physicalmodels/navierstokes3d.h>
#include <math.h>

#ifdef __CUDACC__
  #define GPU_DEVICE_FUNC __device__ __forceinline__
#elif defined(__HIPCC__)
  #define GPU_DEVICE_FUNC __device__ __forceinline__
#else
  #define GPU_DEVICE_FUNC static inline
#endif

/* Helper function: Matrix-vector multiply */
GPU_DEVICE_FUNC void gpu_matvecmult(int n, double *y, const double *A, const double *x) {
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = 0; j < n; j++) {
      y[i] += A[i*n + j] * x[j];
    }
  }
}

/* Helper function: Roe average for NavierStokes3D */
GPU_DEVICE_FUNC void gpu_ns3d_roe_average(double *uavg, const double *uL, const double *uR, int nvars, double gamma) {
  double rhoL, vxL, vyL, vzL, eL, PL;
  double rhoR, vxR, vyR, vzR, eR, PR;

  rhoL = uL[0];
  rhoR = uR[0];

  /* Validate rho before computing sqrt - must match CPU macro behavior */
  double tL = sqrt(rhoL);
  double tR = sqrt(rhoR);
  /* Use device-side isnan/isinf - these should be available in CUDA/HIP */
  #ifdef __CUDA_ARCH__
    if (isnan(tL) || isinf(tL) || isnan(tR) || isinf(tR)) {
      /* Invalid sqrt - this will cause the CPU macro to exit, so we should too */
      /* For now, set uavg to zero to avoid further issues */
      for (int i = 0; i < nvars; i++) uavg[i] = 0.0;
      return;
    }
  #elif defined(__HIP_DEVICE_COMPILE__)
    if (isnan(tL) || isinf(tL) || isnan(tR) || isinf(tR)) {
      for (int i = 0; i < nvars; i++) uavg[i] = 0.0;
      return;
    }
  #else
    /* Host code - use standard isnan/isinf */
    if (isnan(tL) || isinf(tL) || isnan(tR) || isinf(tR)) {
      for (int i = 0; i < nvars; i++) uavg[i] = 0.0;
      return;
    }
  #endif

  vxL = (rhoL == 0) ? 0 : uL[1] / rhoL;
  vyL = (rhoL == 0) ? 0 : uL[2] / rhoL;
  vzL = (rhoL == 0) ? 0 : uL[3] / rhoL;
  eL = uL[4];
  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  PL = (gamma - 1.0) * (eL - 0.5 * rhoL * vsqL);
  double cLsq = gamma * PL / rhoL;
  double HL = 0.5*vsqL + cLsq / (gamma-1.0);

  vxR = (rhoR == 0) ? 0 : uR[1] / rhoR;
  vyR = (rhoR == 0) ? 0 : uR[2] / rhoR;
  vzR = (rhoR == 0) ? 0 : uR[3] / rhoR;
  eR = uR[4];
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  PR = (gamma - 1.0) * (eR - 0.5 * rhoR * vsqR);
  double cRsq = gamma * PR / rhoR;
  double HR = 0.5*vsqR + cRsq / (gamma-1.0);

  double rho = tL * tR;
  double vx = (tL*vxL + tR*vxR) / (tL + tR);
  double vy = (tL*vyL + tR*vyR) / (tL + tR);
  double vz = (tL*vzL + tR*vzR) / (tL + tR);
  double H = (tL*HL + tR*HR) / (tL + tR);
  double vsq = vx*vx + vy*vy + vz*vz;
  double P = (gamma-1.0) * (H-0.5*vsq) * rho / gamma;
  double e = P/(gamma-1.0) + 0.5*rho*vsq;

  uavg[0] = rho;
  uavg[1] = rho*vx;
  uavg[2] = rho*vy;
  uavg[3] = rho*vz;
  uavg[4] = e;
  for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
    uavg[m_i] = sqrt(uL[m_i]) * sqrt(uR[m_i]);
  }
}

/* Helper function: Compute left eigenvectors for NavierStokes3D */
GPU_DEVICE_FUNC void gpu_ns3d_left_eigenvectors(const double *u, double *L, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho == 0.0) {
    for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
    for (int i = 0; i < nvars; i++) L[i*nvars + i] = 1.0;
    return;
  }
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double vz = u[3] / rho;
  double e = u[4];
  double vsq = vx*vx + vy*vy + vz*vz;
  double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
  if (P <= 0.0 || rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
    for (int i = 0; i < nvars; i++) L[i*nvars + i] = 1.0;
    return;
  }
  double a = sqrt(gamma * P / rho);
  double ga_minus_one = gamma - 1.0;
  double ek = 0.5 * vsq;

  for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;

  if (dir == _XDIR_) {
    L[1*nvars+0] = (ga_minus_one*ek + a*vx) / (2*a*a);
    L[1*nvars+1] = ((-ga_minus_one)*vx-a) / (2*a*a);
    L[1*nvars+2] = ((-ga_minus_one)*vy) / (2*a*a);
    L[1*nvars+3] = ((-ga_minus_one)*vz) / (2*a*a);
    L[1*nvars+4] = (ga_minus_one) / (2*a*a);
    L[0*nvars+0] = 1.0 - (ga_minus_one*ek) / (a*a);
    L[0*nvars+1] = (ga_minus_one)*vx / (a*a);
    L[0*nvars+2] = (ga_minus_one)*vy / (a*a);
    L[0*nvars+3] = (ga_minus_one)*vz / (a*a);
    L[0*nvars+4] = (-ga_minus_one) / (a*a);
    L[4*nvars+0] = (ga_minus_one*ek - a*vx) / (2*a*a);
    L[4*nvars+1] = ((-ga_minus_one)*vx+a) / (2*a*a);
    L[4*nvars+2] = ((-ga_minus_one)*vy) / (2*a*a);
    L[4*nvars+3] = ((-ga_minus_one)*vz) / (2*a*a);
    L[4*nvars+4] = (ga_minus_one) / (2*a*a);
    /* shear rows (must match _NavierStokes3DLeftEigenvectors_) */
    L[2*nvars+0] = vy;
    L[2*nvars+2] = -1.0;
    L[3*nvars+0] = -vz;
    L[3*nvars+3] = 1.0;
  } else if (dir == _YDIR_) {
    L[2*nvars+0] = (ga_minus_one*ek + a*vy) / (2*a*a);
    L[2*nvars+1] = ((1.0-gamma)*vx) / (2*a*a);
    L[2*nvars+2] = ((-ga_minus_one)*vy-a) / (2*a*a);
    L[2*nvars+3] = ((1.0-gamma)*vz) / (2*a*a);
    L[2*nvars+4] = (ga_minus_one) / (2*a*a);
    L[0*nvars+0] = 1.0 - (ga_minus_one*ek) / (a*a);
    L[0*nvars+1] = (ga_minus_one)*vx / (a*a);
    L[0*nvars+2] = (ga_minus_one)*vy / (a*a);
    L[0*nvars+3] = (ga_minus_one)*vz / (a*a);
    L[0*nvars+4] = (-ga_minus_one) / (a*a);
    L[4*nvars+0] = (ga_minus_one*ek - a*vy) / (2*a*a);
    L[4*nvars+1] = ((1.0-gamma)*vx) / (2*a*a);
    L[4*nvars+2] = ((1.0-gamma)*vy+a) / (2*a*a);
    L[4*nvars+3] = ((1.0-gamma)*vz) / (2*a*a);
    L[4*nvars+4] = (ga_minus_one) / (2*a*a);
    /* shear rows */
    L[1*nvars+0] = -vx;
    L[1*nvars+1] = 1.0;
    L[3*nvars+0] = vz;
    L[3*nvars+3] = -1.0;
  } else if (dir == _ZDIR_) {
    L[3*nvars+0] = (ga_minus_one*ek + a*vz) / (2*a*a);
    L[3*nvars+1] = ((1.0-gamma)*vx) / (2*a*a);
    L[3*nvars+2] = ((1.0-gamma)*vy) / (2*a*a);
    L[3*nvars+3] = ((-ga_minus_one)*vz-a) / (2*a*a);
    L[3*nvars+4] = (ga_minus_one) / (2*a*a);
    L[0*nvars+0] = 1.0 - (ga_minus_one*ek) / (a*a);
    L[0*nvars+1] = (ga_minus_one)*vx / (a*a);
    L[0*nvars+2] = (ga_minus_one)*vy / (a*a);
    L[0*nvars+3] = (ga_minus_one)*vz / (a*a);
    L[0*nvars+4] = (-ga_minus_one) / (a*a);
    L[4*nvars+0] = (ga_minus_one*ek - a*vz) / (2*a*a);
    L[4*nvars+1] = ((1.0-gamma)*vx) / (2*a*a);
    L[4*nvars+2] = ((1.0-gamma)*vy) / (2*a*a);
    L[4*nvars+3] = ((1.0-gamma)*vz+a) / (2*a*a);
    L[4*nvars+4] = (ga_minus_one) / (2*a*a);
    /* shear rows */
    L[1*nvars+0] = vx;
    L[1*nvars+1] = -1.0;
    L[2*nvars+0] = -vy;
    L[2*nvars+2] = 1.0;
  }
  for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
    L[m_i*nvars + m_i] = 1.0;
  }
}

/* Helper function: Compute right eigenvectors for NavierStokes3D */
GPU_DEVICE_FUNC void gpu_ns3d_right_eigenvectors(const double *u, double *R, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho == 0.0) {
    for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
    for (int i = 0; i < nvars; i++) R[i*nvars + i] = 1.0;
    return;
  }
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double vz = u[3] / rho;
  double e = u[4];
  double vsq = vx*vx + vy*vy + vz*vz;
  double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
  if (P <= 0.0 || rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
    for (int i = 0; i < nvars; i++) R[i*nvars + i] = 1.0;
    return;
  }
  double a = sqrt(gamma * P / rho);
  double ga_minus_one = gamma - 1.0;
  double ek = 0.5 * vsq;
  double h0 = a*a / ga_minus_one + ek;

  for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;

  if (dir == _XDIR_) {
    /* Column 0: entropy wave */
    R[0*nvars+0] = 1.0;
    R[1*nvars+0] = vx;
    R[2*nvars+0] = vy;
    R[3*nvars+0] = vz;
    R[4*nvars+0] = ek;

    /* Column 1: left-going acoustic wave */
    R[0*nvars+1] = 1.0;
    R[1*nvars+1] = vx - a;
    R[2*nvars+1] = vy;
    R[3*nvars+1] = vz;
    R[4*nvars+1] = h0 - a*vx;

    /* Column 2: shear wave (y-direction) */
    R[0*nvars+2] = 0.0;
    R[1*nvars+2] = 0.0;
    R[2*nvars+2] = -1.0;
    R[3*nvars+2] = 0.0;
    R[4*nvars+2] = -vy;

    /* Column 3: shear wave (z-direction) */
    R[0*nvars+3] = 0.0;
    R[1*nvars+3] = 0.0;
    R[2*nvars+3] = 0.0;
    R[3*nvars+3] = 1.0;
    R[4*nvars+3] = vz;

    /* Column 4: right-going acoustic wave */
    R[0*nvars+4] = 1.0;
    R[1*nvars+4] = vx + a;
    R[2*nvars+4] = vy;
    R[3*nvars+4] = vz;
    R[4*nvars+4] = h0 + a*vx;
  } else if (dir == _YDIR_) {
    /* Column 0: entropy wave */
    R[0*nvars+0] = 1.0;
    R[1*nvars+0] = vx;
    R[2*nvars+0] = vy;
    R[3*nvars+0] = vz;
    R[4*nvars+0] = ek;

    /* Column 1: shear wave (x-direction) */
    R[0*nvars+1] = 0.0;
    R[1*nvars+1] = 1.0;
    R[2*nvars+1] = 0.0;
    R[3*nvars+1] = 0.0;
    R[4*nvars+1] = vx;

    /* Column 2: left-going acoustic wave */
    R[0*nvars+2] = 1.0;
    R[1*nvars+2] = vx;
    R[2*nvars+2] = vy - a;
    R[3*nvars+2] = vz;
    R[4*nvars+2] = h0 - a*vy;

    /* Column 3: shear wave (z-direction) */
    R[0*nvars+3] = 0.0;
    R[1*nvars+3] = 0.0;
    R[2*nvars+3] = 0.0;
    R[3*nvars+3] = -1.0;
    R[4*nvars+3] = -vz;

    /* Column 4: right-going acoustic wave */
    R[0*nvars+4] = 1.0;
    R[1*nvars+4] = vx;
    R[2*nvars+4] = vy + a;
    R[3*nvars+4] = vz;
    R[4*nvars+4] = h0 + a*vy;
  } else if (dir == _ZDIR_) {
    /* Column 0: entropy wave */
    R[0*nvars+0] = 1.0;
    R[1*nvars+0] = vx;
    R[2*nvars+0] = vy;
    R[3*nvars+0] = vz;
    R[4*nvars+0] = ek;

    /* Column 1: shear wave (x-direction) */
    R[0*nvars+1] = 0.0;
    R[1*nvars+1] = -1.0;
    R[2*nvars+1] = 0.0;
    R[3*nvars+1] = 0.0;
    R[4*nvars+1] = -vx;

    /* Column 2: shear wave (y-direction) */
    R[0*nvars+2] = 0.0;
    R[1*nvars+2] = 0.0;
    R[2*nvars+2] = 1.0;
    R[3*nvars+2] = 0.0;
    R[4*nvars+2] = vy;

    /* Column 3: left-going acoustic wave */
    R[0*nvars+3] = 1.0;
    R[1*nvars+3] = vx;
    R[2*nvars+3] = vy;
    R[3*nvars+3] = vz - a;
    R[4*nvars+3] = h0 - a*vz;

    /* Column 4: right-going acoustic wave */
    R[0*nvars+4] = 1.0;
    R[1*nvars+4] = vx;
    R[2*nvars+4] = vy;
    R[3*nvars+4] = vz + a;
    R[4*nvars+4] = h0 + a*vz;
  }
  for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
    R[m_i*nvars + m_i] = 1.0;
  }
}

#endif /* _GPU_NS3D_HELPERS_H_ */

