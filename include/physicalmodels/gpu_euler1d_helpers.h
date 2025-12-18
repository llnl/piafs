/*! @file gpu_euler1d_helpers.h
    @brief Shared GPU helper functions for Euler1D
    @details These are inline device functions that can be included in multiple .cu files
*/

#ifndef _GPU_EULER1D_HELPERS_H_
#define _GPU_EULER1D_HELPERS_H_

#include <physicalmodels/euler1d.h>
#include <math.h>

#ifdef __CUDACC__
  #define GPU_DEVICE_FUNC __device__ __forceinline__
#elif defined(__HIPCC__)
  #define GPU_DEVICE_FUNC __device__ __forceinline__
#else
  #define GPU_DEVICE_FUNC static inline
#endif

/* Helper function: Roe average for Euler1D
   Matches _Euler1DRoeAverage_ macro from euler1d.h */
GPU_DEVICE_FUNC void gpu_euler1d_roe_average(double *uavg, const double *uL, const double *uR, int nvars, double gamma) {
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
  
  double vL = (rhoL == 0) ? 0 : uL[1] / rhoL;
  double eL = uL[2];
  double PL = (eL - 0.5*rhoL*vL*vL) * (gamma - 1.0);
  double cLsq = gamma * PL / rhoL;
  double HL = 0.5*vL*vL + cLsq / (gamma - 1.0);
  
  double vR = (rhoR == 0) ? 0 : uR[1] / rhoR;
  double eR = uR[2];
  double PR = (eR - 0.5*rhoR*vR*vR) * (gamma - 1.0);
  double cRsq = gamma * PR / rhoR;
  double HR = 0.5*vR*vR + cRsq / (gamma - 1.0);
  
  double rho = tL * tR;
  double v = (tL*vL + tR*vR) / (tL + tR);
  double H = (tL*HL + tR*HR) / (tL + tR);
  double csq = (gamma - 1.0) * (H - 0.5*v*v);
  double P = csq * rho / gamma;
  double e = P / (gamma - 1.0) + 0.5*rho*v*v;
  
  uavg[0] = rho;
  uavg[1] = rho * v;
  uavg[2] = e;
  
  /* Passive scalars */
  for (int m_i = _EU1D_NVARS_; m_i < nvars; m_i++) {
    uavg[m_i] = sqrt(uL[m_i]) * sqrt(uR[m_i]);
  }
}

/* Helper function: Compute left eigenvectors for Euler1D
   Matches _Euler1DLeftEigenvectors_ macro from euler1d.h
   Note: The macro has the ordering: row 0 = entropy, row 1 = left-acoustic, row 2 = right-acoustic */
GPU_DEVICE_FUNC void gpu_euler1d_left_eigenvectors(const double *u, double *L, double gamma, int nvars) {
  double rho = u[0];
  if (rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
    for (int i = 0; i < nvars; i++) L[i*nvars + i] = 1.0;
    return;
  }
  
  double v = u[1] / rho;
  double e = u[2];
  double P = (e - 0.5*rho*v*v) * (gamma - 1.0);
  
  if (P <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
    for (int i = 0; i < nvars; i++) L[i*nvars + i] = 1.0;
    return;
  }
  
  double c = sqrt(gamma * P / rho);
  double ga_m1 = gamma - 1.0;
  
  /* Initialize to zero */
  for (int i = 0; i < nvars*nvars; i++) L[i] = 0.0;
  
  /* From _Euler1DLeftEigenvectors_ macro:
     L[1*nvars+0] = ((gamma - 1)/(rho*c)) * (-(v*v)/2 - c*v/(gamma-1));
     L[1*nvars+1] = ((gamma - 1)/(rho*c)) * (v + c/(gamma-1));
     L[1*nvars+2] = ((gamma - 1)/(rho*c)) * (-1);
     L[0*nvars+0] = ((gamma - 1)/(rho*c)) * (rho*(-(v*v)/2+c*c/(gamma-1))/c);
     L[0*nvars+1] = ((gamma - 1)/(rho*c)) * (rho*v/c);
     L[0*nvars+2] = ((gamma - 1)/(rho*c)) * (-rho/c);
     L[2*nvars+0] = ((gamma - 1)/(rho*c)) * ((v*v)/2 - c*v/(gamma-1));
     L[2*nvars+1] = ((gamma - 1)/(rho*c)) * (-v + c/(gamma-1));
     L[2*nvars+2] = ((gamma - 1)/(rho*c)) * (1);
  */
  double factor = ga_m1 / (rho * c);
  
  /* Row 1: left-going acoustic wave */
  L[1*nvars+0] = factor * (-(v*v)/2 - c*v/ga_m1);
  L[1*nvars+1] = factor * (v + c/ga_m1);
  L[1*nvars+2] = factor * (-1.0);
  
  /* Row 0: entropy wave */
  L[0*nvars+0] = factor * (rho*(-(v*v)/2 + c*c/ga_m1)/c);
  L[0*nvars+1] = factor * (rho*v/c);
  L[0*nvars+2] = factor * (-rho/c);
  
  /* Row 2: right-going acoustic wave */
  L[2*nvars+0] = factor * ((v*v)/2 - c*v/ga_m1);
  L[2*nvars+1] = factor * (-v + c/ga_m1);
  L[2*nvars+2] = factor * (1.0);
  
  /* Passive scalars: identity */
  for (int m_i = _EU1D_NVARS_; m_i < nvars; m_i++) {
    L[m_i*nvars + m_i] = 1.0;
  }
}

/* Helper function: Compute right eigenvectors for Euler1D
   Matches _Euler1DRightEigenvectors_ macro from euler1d.h */
GPU_DEVICE_FUNC void gpu_euler1d_right_eigenvectors(const double *u, double *R, double gamma, int nvars) {
  double rho = u[0];
  if (rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
    for (int i = 0; i < nvars; i++) R[i*nvars + i] = 1.0;
    return;
  }
  
  double v = u[1] / rho;
  double e = u[2];
  double P = (e - 0.5*rho*v*v) * (gamma - 1.0);
  
  if (P <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
    for (int i = 0; i < nvars; i++) R[i*nvars + i] = 1.0;
    return;
  }
  
  double c = sqrt(gamma * P / rho);
  double ga_m1 = gamma - 1.0;
  
  /* Initialize to zero */
  for (int i = 0; i < nvars*nvars; i++) R[i] = 0.0;
  
  /* From _Euler1DRightEigenvectors_ macro:
     R[0*nvars+1] = - rho/(2*c);  R[1*nvars+1] = -rho*(v-c)/(2*c); R[2*nvars+1] = -rho*((v*v)/2+(c*c)/(gamma-1)-c*v)/(2*c);
     R[0*nvars+0] = 1;            R[1*nvars+0] = v;                R[2*nvars+0] = v*v / 2;
     R[0*nvars+2] = rho/(2*c);    R[1*nvars+2] = rho*(v+c)/(2*c);  R[2*nvars+2] = rho*((v*v)/2+(c*c)/(gamma-1)+c*v)/(2*c);
  */
  
  /* Column 0: entropy wave */
  R[0*nvars+0] = 1.0;
  R[1*nvars+0] = v;
  R[2*nvars+0] = v*v / 2.0;
  
  /* Column 1: left-going acoustic wave */
  R[0*nvars+1] = -rho / (2*c);
  R[1*nvars+1] = -rho*(v - c) / (2*c);
  R[2*nvars+1] = -rho*((v*v)/2 + (c*c)/ga_m1 - c*v) / (2*c);
  
  /* Column 2: right-going acoustic wave */
  R[0*nvars+2] = rho / (2*c);
  R[1*nvars+2] = rho*(v + c) / (2*c);
  R[2*nvars+2] = rho*((v*v)/2 + (c*c)/ga_m1 + c*v) / (2*c);
  
  /* Passive scalars: identity */
  for (int m_i = _EU1D_NVARS_; m_i < nvars; m_i++) {
    R[m_i*nvars + m_i] = 1.0;
  }
}

#endif /* _GPU_EULER1D_HELPERS_H_ */

