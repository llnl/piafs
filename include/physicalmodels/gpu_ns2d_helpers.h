/*! @file gpu_ns2d_helpers.h
    @brief Shared GPU helper functions for NavierStokes2D
    @details These are inline device functions that can be included in multiple .cu files
*/

#ifndef _GPU_NS2D_HELPERS_H_
#define _GPU_NS2D_HELPERS_H_

#include <physicalmodels/navierstokes2d.h>
#include <math.h>

#ifdef __CUDACC__
  #define GPU_DEVICE_FUNC __device__ __forceinline__
#elif defined(__HIPCC__)
  #define GPU_DEVICE_FUNC __device__ __forceinline__
#else
  #define GPU_DEVICE_FUNC static inline
#endif

/* Helper function: Roe average for NavierStokes2D
   Matches _NavierStokes2DRoeAverage_ macro from navierstokes2d.h */
GPU_DEVICE_FUNC void gpu_ns2d_roe_average(double *uavg, const double *uL, const double *uR, int nvars, double gamma) {
  double rhoL = uL[0];
  double rhoR = uR[0];
  
  double tL = sqrt(rhoL);
  double tR = sqrt(rhoR);
  
  /* Check for invalid sqrt */
  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (isnan(tL) || isinf(tL) || isnan(tR) || isinf(tR)) {
      for (int i = 0; i < nvars; i++) uavg[i] = 0.0;
      return;
    }
  #else
    if (isnan(tL) || isinf(tL) || isnan(tR) || isinf(tR)) {
      for (int i = 0; i < nvars; i++) uavg[i] = 0.0;
      return;
    }
  #endif
  
  double vxL = (rhoL == 0) ? 0 : uL[1] / rhoL;
  double vyL = (rhoL == 0) ? 0 : uL[2] / rhoL;
  double eL = uL[3];
  double vsqL = vxL*vxL + vyL*vyL;
  double PL = (eL - 0.5*rhoL*vsqL) * (gamma - 1.0);
  double cLsq = gamma * PL / rhoL;
  double HL = 0.5*vsqL + cLsq / (gamma - 1.0);
  
  double vxR = (rhoR == 0) ? 0 : uR[1] / rhoR;
  double vyR = (rhoR == 0) ? 0 : uR[2] / rhoR;
  double eR = uR[3];
  double vsqR = vxR*vxR + vyR*vyR;
  double PR = (eR - 0.5*rhoR*vsqR) * (gamma - 1.0);
  double cRsq = gamma * PR / rhoR;
  double HR = 0.5*vsqR + cRsq / (gamma - 1.0);
  
  double rho = tL * tR;
  double vx = (tL*vxL + tR*vxR) / (tL + tR);
  double vy = (tL*vyL + tR*vyR) / (tL + tR);
  double H = (tL*HL + tR*HR) / (tL + tR);
  double vsq = vx*vx + vy*vy;
  double csq = (gamma - 1.0) * (H - 0.5*vsq);
  double P = csq * rho / gamma;
  double e = P / (gamma - 1.0) + 0.5*rho*vsq;
  
  uavg[0] = rho;
  uavg[1] = rho * vx;
  uavg[2] = rho * vy;
  uavg[3] = e;
  
  /* Passive scalars */
  for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
    uavg[m_i] = sqrt(uL[m_i]) * sqrt(uR[m_i]);
  }
}

/* Helper function: Compute left eigenvectors for NavierStokes2D
   Matches _NavierStokes2DLeftEigenvectors_ macro from navierstokes2d.h */
GPU_DEVICE_FUNC void gpu_ns2d_left_eigenvectors(const double *u, double *L, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
    for (int i = 0; i < nvars; i++) L[i*nvars + i] = 1.0;
    return;
  }
  
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double e = u[3];
  double vsq = vx*vx + vy*vy;
  double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
  
  if (P <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
    for (int i = 0; i < nvars; i++) L[i*nvars + i] = 1.0;
    return;
  }
  
  double a = sqrt(gamma * P / rho);
  double ga_m1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double nx = 0.0, ny = 0.0;
  double un;
  
  /* Initialize to zero */
  for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
  
  if (dir == _XDIR_) {
    un = vx;
    nx = 1.0;
    L[0*nvars+0] = (ga_m1*ek + a*un) / (2*a*a);
    L[0*nvars+1] = ((-ga_m1)*vx - a*nx) / (2*a*a);
    L[0*nvars+2] = ((-ga_m1)*vy - a*ny) / (2*a*a);
    L[0*nvars+3] = ga_m1 / (2*a*a);
    L[3*nvars+0] = (a*a - ga_m1*ek) / (a*a);
    L[3*nvars+1] = (ga_m1*vx) / (a*a);
    L[3*nvars+2] = (ga_m1*vy) / (a*a);
    L[3*nvars+3] = (-ga_m1) / (a*a);
    L[1*nvars+0] = (ga_m1*ek - a*un) / (2*a*a);
    L[1*nvars+1] = ((-ga_m1)*vx + a*nx) / (2*a*a);
    L[1*nvars+2] = ((-ga_m1)*vy + a*ny) / (2*a*a);
    L[1*nvars+3] = ga_m1 / (2*a*a);
    L[2*nvars+0] = (vy - un*ny) / nx;
    L[2*nvars+1] = ny;
    L[2*nvars+2] = (ny*ny - 1.0) / nx;
    L[2*nvars+3] = 0.0;
  } else if (dir == _YDIR_) {
    un = vy;
    ny = 1.0;
    L[0*nvars+0] = (ga_m1*ek + a*un) / (2*a*a);
    L[0*nvars+1] = ((1.0-gamma)*vx - a*nx) / (2*a*a);
    L[0*nvars+2] = ((1.0-gamma)*vy - a*ny) / (2*a*a);
    L[0*nvars+3] = ga_m1 / (2*a*a);
    L[3*nvars+0] = (a*a - ga_m1*ek) / (a*a);
    L[3*nvars+1] = ga_m1*vx / (a*a);
    L[3*nvars+2] = ga_m1*vy / (a*a);
    L[3*nvars+3] = (1.0 - gamma) / (a*a);
    L[2*nvars+0] = (ga_m1*ek - a*un) / (2*a*a);
    L[2*nvars+1] = ((1.0-gamma)*vx + a*nx) / (2*a*a);
    L[2*nvars+2] = ((1.0-gamma)*vy + a*ny) / (2*a*a);
    L[2*nvars+3] = ga_m1 / (2*a*a);
    L[1*nvars+0] = (un*nx - vx) / ny;
    L[1*nvars+1] = (1.0 - nx*nx) / ny;
    L[1*nvars+2] = -nx;
    L[1*nvars+3] = 0.0;
  }
  
  /* Passive scalars: identity */
  for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
    L[m_i*nvars + m_i] = 1.0;
  }
}

/* Helper function: Compute right eigenvectors for NavierStokes2D
   Matches _NavierStokes2DRightEigenvectors_ macro from navierstokes2d.h */
GPU_DEVICE_FUNC void gpu_ns2d_right_eigenvectors(const double *u, double *R, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
    for (int i = 0; i < nvars; i++) R[i*nvars + i] = 1.0;
    return;
  }
  
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double e = u[3];
  double vsq = vx*vx + vy*vy;
  double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
  
  if (P <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
    for (int i = 0; i < nvars; i++) R[i*nvars + i] = 1.0;
    return;
  }
  
  double a = sqrt(gamma * P / rho);
  double ga_m1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double h0 = a*a / ga_m1 + ek;
  double nx = 0.0, ny = 0.0;
  double un;
  
  /* Initialize to zero */
  for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
  
  if (dir == _XDIR_) {
    un = vx;
    nx = 1.0;
    R[0*nvars+0] = 1.0;
    R[1*nvars+0] = vx - a*nx;
    R[2*nvars+0] = vy - a*ny;
    R[3*nvars+0] = h0 - a*un;
    R[0*nvars+3] = 1.0;
    R[1*nvars+3] = vx;
    R[2*nvars+3] = vy;
    R[3*nvars+3] = ek;
    R[0*nvars+1] = 1.0;
    R[1*nvars+1] = vx + a*nx;
    R[2*nvars+1] = vy + a*ny;
    R[3*nvars+1] = h0 + a*un;
    R[0*nvars+2] = 0.0;
    R[1*nvars+2] = ny;
    R[2*nvars+2] = -nx;
    R[3*nvars+2] = vx*ny - vy*nx;
  } else if (dir == _YDIR_) {
    un = vy;
    ny = 1.0;
    R[0*nvars+0] = 1.0;
    R[1*nvars+0] = vx - a*nx;
    R[2*nvars+0] = vy - a*ny;
    R[3*nvars+0] = h0 - a*un;
    R[0*nvars+3] = 1.0;
    R[1*nvars+3] = vx;
    R[2*nvars+3] = vy;
    R[3*nvars+3] = ek;
    R[0*nvars+2] = 1.0;
    R[1*nvars+2] = vx + a*nx;
    R[2*nvars+2] = vy + a*ny;
    R[3*nvars+2] = h0 + a*un;
    R[0*nvars+1] = 0.0;
    R[1*nvars+1] = ny;
    R[2*nvars+1] = -nx;
    R[3*nvars+1] = vx*ny - vy*nx;
  }
  
  /* Passive scalars: identity */
  for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
    R[m_i*nvars + m_i] = 1.0;
  }
}

#endif /* _GPU_NS2D_HELPERS_H_ */

