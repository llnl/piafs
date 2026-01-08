/*! @file gpu_upwind.cu
    @brief GPU kernels for NavierStokes3D upwinding
*/

#include <gpu.h>
#include <physicalmodels/gpu_ns3d_helpers.h>
#include <math.h>

/* Compile-time constant for 3D - enables loop unrolling and better optimization */
#define NS3D_NDIMS 3

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Helper function: Compute absolute value */
__device__ double gpu_absolute(double x) {
  return (x < 0.0) ? -x : x;
}

/* Helper function: Matrix multiply */
__device__ void gpu_matmult(int n, double *C, const double *A, const double *B) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i*n + j] = 0.0;
      for (int k = 0; k < n; k++) {
        C[i*n + j] += A[i*n + k] * B[k*n + j];
      }
    }
  }
}

/* Helper function: Compute eigenvalues */
__device__ void gpu_ns3d_eigenvalues(const double *u, double *D, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho == 0.0) {
    /* Handle zero density case */
    for (int i = 0; i < nvars*nvars; i++) D[i] = 0.0;
    return;
  }
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double vz = u[3] / rho;
  double e = u[4];
  double vsq = vx*vx + vy*vy + vz*vz;
  double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
  if (P <= 0.0 || rho <= 0.0) {
    /* Handle invalid state */
    for (int i = 0; i < nvars*nvars; i++) D[i] = 0.0;
    return;
  }
  double c = sqrt(gamma * P / rho);
  
  double vn;
  if (dir == _XDIR_) vn = vx;
  else if (dir == _YDIR_) vn = vy;
  else if (dir == _ZDIR_) vn = vz;
  else vn = 0.0;
  
  /* Initialize D to zero */
  for (int i = 0; i < nvars*nvars; i++) D[i] = 0.0;
  
  D[0*nvars+0] = vn;
  if (dir == _XDIR_) {
    D[1*nvars+1] = vn - c;
    D[2*nvars+2] = vn;
    D[3*nvars+3] = vn;
  } else if (dir == _YDIR_) {
    D[1*nvars+1] = vn;
    D[2*nvars+2] = vn - c;
    D[3*nvars+3] = vn;
  } else if (dir == _ZDIR_) {
    D[1*nvars+1] = vn;
    D[2*nvars+2] = vn;
    D[3*nvars+3] = vn - c;
  }
  D[4*nvars+4] = vn + c;
  for (int m_i = _NS3D_NVARS_; m_i < nvars; m_i++) {
    D[m_i*nvars+m_i] = vn;
  }
}

/* Kernel: Roe upwinding for NavierStokes3D
   Each thread handles one interface point
*/
GPU_KERNEL void gpu_ns3d_upwind_roe_kernel(
  double *fI,              /* output: upwind interface flux */
  const double *fL,        /* input: left-biased flux */
  const double *fR,        /* input: right-biased flux */
  const double *uL,        /* input: left-biased solution */
  const double *uR,        /* input: right-biased solution */
  const double *u,          /* input: cell-centered solution */
  int nvars,               /* number of variables */
  int ndims,                /* number of dimensions */
  const int *dim,          /* dimension sizes (without ghosts) */
  const int *stride_with_ghosts, /* stride array */
  const int *bounds_inter, /* bounds for interface array */
  int ghosts,              /* number of ghost points */
  int dir,                 /* direction */
  double gamma,            /* gamma parameter */
  double *workspace        /* dynamically allocated workspace */
)
{
  /* Compute total number of interface points - unrolled for 3D */
  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    /* Decompose idx into 3D interface index - unrolled for compiler optimization */
    int indexI[NS3D_NDIMS];
    int temp = idx;
    indexI[2] = temp % bounds_inter[2]; temp /= bounds_inter[2];
    indexI[1] = temp % bounds_inter[1]; temp /= bounds_inter[1];
    indexI[0] = temp;
    
    /* Compute 1D index for interface point (no ghosts) - unrolled */
    int p = indexI[2] + bounds_inter[2] * (indexI[1] + bounds_inter[1] * indexI[0]);
    
    /* Compute cell indices for left and right cells - unrolled */
    int indexL[NS3D_NDIMS] = {indexI[0], indexI[1], indexI[2]};
    int indexR[NS3D_NDIMS] = {indexI[0], indexI[1], indexI[2]};
    indexL[dir]--;
    
    /* Compute 1D indices for cell-centered arrays (with ghosts) - unrolled */
    int pL = (indexL[0] + ghosts) * stride_with_ghosts[0] +
             (indexL[1] + ghosts) * stride_with_ghosts[1] +
             (indexL[2] + ghosts) * stride_with_ghosts[2];
    int pR = (indexR[0] + ghosts) * stride_with_ghosts[0] +
             (indexR[1] + ghosts) * stride_with_ghosts[1] +
             (indexR[2] + ghosts) * stride_with_ghosts[2];
    
    /* Roe's upwinding scheme - use dynamic workspace */
    /* Workspace layout: [udiff, uavg, udiss, D, L, R, DL, modA] */
    /* Total: 3*nvars + 5*nvars*nvars per thread */
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t workspace_per_thread = 3 * nvars + 5 * nvars * nvars;
    double *thread_workspace = workspace + threadId * workspace_per_thread;
    
    double *udiff = thread_workspace;
    double *uavg = udiff + nvars;
    double *udiss = uavg + nvars;
    double *D = udiss + nvars;
    double *L = D + nvars * nvars;
    double *R = L + nvars * nvars;
    double *DL = R + nvars * nvars;
    double *modA = DL + nvars * nvars;
    
    for (int k = 0; k < nvars; k++) {
      udiff[k] = 0.5 * (uR[p*nvars+k] - uL[p*nvars+k]);
    }
    
    /* Compute Roe average */
    gpu_ns3d_roe_average(uavg, u + pL*nvars, u + pR*nvars, nvars, gamma);
    
    /* Compute eigenvalues, left and right eigenvectors */
    
    gpu_ns3d_eigenvalues(uavg, D, gamma, nvars, dir);
    gpu_ns3d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns3d_right_eigenvectors(uavg, R, gamma, nvars, dir);
    
    /* Harten's Entropy Fix */
    double delta = 0.000001, delta2 = delta*delta;
    for (int k = 0; k < nvars; k++) {
      double absD = gpu_absolute(D[k*nvars+k]);
      D[k*nvars+k] = (absD < delta) ? (D[k*nvars+k]*D[k*nvars+k]+delta2)/(2*delta) : absD;
    }
    
    /* Compute modA = R * |D| * L */
    gpu_matmult(nvars, DL, D, L);
    gpu_matmult(nvars, modA, R, DL);
    
    /* Compute dissipation: modA * udiff */
    gpu_matvecmult(nvars, udiss, modA, udiff);
    
    /* Compute upwind flux */
    for (int k = 0; k < nvars; k++) {
      fI[p*nvars+k] = 0.5 * (fL[p*nvars+k] + fR[p*nvars+k]) - udiss[k];
    }
  }
}

/* Helper function: max3 */
__device__ double gpu_max3(double a, double b, double c) {
  double ab = (a > b) ? a : b;
  return (ab > c) ? ab : c;
}

/* Helper: Optimized matvec multiply for nvars=5 - fully unrolled */
__device__ void gpu_matvecmult_5(double *y, const double *A, const double *x) {
  #pragma unroll
  for (int i = 0; i < 5; i++) {
    y[i] = A[i*5+0]*x[0] + A[i*5+1]*x[1] + A[i*5+2]*x[2] + A[i*5+3]*x[3] + A[i*5+4]*x[4];
  }
}

/* Helper: Optimized matvec multiply for nvars=12 - fully unrolled */
__device__ void gpu_matvecmult_12(double *y, const double *A, const double *x) {
  #pragma unroll
  for (int i = 0; i < 12; i++) {
    y[i] = A[i*12+0]*x[0] + A[i*12+1]*x[1] + A[i*12+2]*x[2] + A[i*12+3]*x[3] +
           A[i*12+4]*x[4] + A[i*12+5]*x[5] + A[i*12+6]*x[6] + A[i*12+7]*x[7] +
           A[i*12+8]*x[8] + A[i*12+9]*x[9] + A[i*12+10]*x[10] + A[i*12+11]*x[11];
  }
}

/* Helper: Extract eigenvalues directly without full matrix - optimized for NS3D */
__device__ void gpu_ns3d_eigenvalues_diag(const double *u, double *eig, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho == 0.0) {
    #pragma unroll
    for (int i = 0; i < 12; i++) eig[i] = 0.0;  /* Max nvars we optimize for */
    return;
  }
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double vz = u[3] / rho;
  double e = u[4];
  double vsq = vx*vx + vy*vy + vz*vz;
  double P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
  if (P <= 0.0 || rho <= 0.0) {
    #pragma unroll
    for (int i = 0; i < 12; i++) eig[i] = 0.0;
    return;
  }
  double c = sqrt(gamma * P / rho);
  
  double vn = (dir == _XDIR_) ? vx : ((dir == _YDIR_) ? vy : vz);
  
  /* Extract diagonal eigenvalues directly - pattern depends on direction */
  eig[0] = vn;
  if (dir == _XDIR_) {
    eig[1] = vn - c;
    eig[2] = vn;
    eig[3] = vn;
  } else if (dir == _YDIR_) {
    eig[1] = vn;
    eig[2] = vn - c;
    eig[3] = vn;
  } else {  /* _ZDIR_ */
    eig[1] = vn;
    eig[2] = vn;
    eig[3] = vn - c;
  }
  eig[4] = vn + c;
  /* Passively advected species have eigenvalue vn */
  #pragma unroll
  for (int m_i = 5; m_i < nvars; m_i++) {
    eig[m_i] = vn;
  }
}

/* Kernel: RF upwinding optimized for nvars=5 */
GPU_KERNEL void gpu_ns3d_upwind_rf_kernel_nvars5(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  /* Compile-time constants for optimization */
  const int nvars = 5;
  
  /* Unrolled for 3D */
  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    /* Decompose idx - unrolled for 3D */
    int indexI[3];
    int temp = idx;
    indexI[2] = temp % bounds_inter[2]; temp /= bounds_inter[2];
    indexI[1] = temp % bounds_inter[1]; temp /= bounds_inter[1];
    indexI[0] = temp;
    
    /* Compute 1D interface index - unrolled */
    int p = indexI[2] + bounds_inter[2] * (indexI[1] + bounds_inter[1] * indexI[0]);
    
    /* Compute cell indices - unrolled */
    int indexL[3] = {indexI[0], indexI[1], indexI[2]};
    indexL[dir]--;

    /* Use shared memory for thread-local workspace */
    double uavg[5], fcL[5], fcR[5], ucL[5], ucR[5], fc[5];
    double eigL[5], eigC[5], eigR[5];
    double L[25], R[25];  /* 5x5 matrices */
    
    /* Roe average */
    gpu_ns3d_roe_average(uavg, uL + p*nvars, uR + p*nvars, nvars, gamma);
    
    /* Eigenvectors */
    gpu_ns3d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns3d_right_eigenvectors(uavg, R, gamma, nvars, dir);
    
    /* Matrix-vector multiplies - use optimized version */
    gpu_matvecmult_5(ucL, L, uL + p*nvars);
    gpu_matvecmult_5(ucR, L, uR + p*nvars);
    gpu_matvecmult_5(fcL, L, fL + p*nvars);
    gpu_matvecmult_5(fcR, L, fR + p*nvars);
    
    /* Eigenvalues - extract diagonal only */
    gpu_ns3d_eigenvalues_diag(uL + p*nvars, eigL, gamma, nvars, dir);
    gpu_ns3d_eigenvalues_diag(uR + p*nvars, eigR, gamma, nvars, dir);
    gpu_ns3d_eigenvalues_diag(uavg, eigC, gamma, nvars, dir);
    
    /* Compute characteristic fluxes - fully unrolled */
    #pragma unroll
    for (int k = 0; k < 5; k++) {
      if ((eigL[k] > 0) && (eigC[k] > 0) && (eigR[k] > 0)) {
        fc[k] = fcL[k];
      } else if ((eigL[k] < 0) && (eigC[k] < 0) && (eigR[k] < 0)) {
        fc[k] = fcR[k];
      } else {
        double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
        fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
      }
    }
    
    /* Transform back - use optimized version */
    gpu_matvecmult_5(fI + p*nvars, R, fc);
  }
}

/* Kernel: RF upwinding optimized for nvars=12
 * Key optimization: Only compute 5x5 eigenvector block for base flow variables.
 * Passive scalars (vars 5-11) use identity eigenvectors (simple upwinding).
 * This reduces register usage and avoids global memory workspace.
 */
GPU_KERNEL void gpu_ns3d_upwind_rf_kernel_nvars12(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  const int nvars = 12;
  /* Note: Only 5 base flow variables need characteristic decomposition;
     passive scalars (vars 5-11) use simple upwinding with identity eigenvectors */

  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_interfaces) return;

  /* Decompose idx - unrolled for 3D */
  int temp = idx;
  const int i2 = temp % bounds_inter[2]; temp /= bounds_inter[2];
  const int i1 = temp % bounds_inter[1]; temp /= bounds_inter[1];
  const int i0 = temp;

  /* Interface index */
  const int p = i2 + bounds_inter[2] * (i1 + bounds_inter[1] * i0);

  /* Pointers to interface data */
  const double *uL_p = uL + p * nvars;
  const double *uR_p = uR + p * nvars;
  const double *fL_p = fL + p * nvars;
  const double *fR_p = fR + p * nvars;
  double *fI_p = fI + p * nvars;

  /* ===== Compute Roe average for base 5 variables ===== */
  double rhoL = uL_p[0], rhoR = uR_p[0];
  double tL = sqrt(rhoL), tR = sqrt(rhoR);
  double tLpR = tL + tR;

  double vxL = uL_p[1]/rhoL, vyL = uL_p[2]/rhoL, vzL = uL_p[3]/rhoL;
  double vxR = uR_p[1]/rhoR, vyR = uR_p[2]/rhoR, vzR = uR_p[3]/rhoR;

  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  double PL = (gamma-1.0) * (uL_p[4] - 0.5*rhoL*vsqL);
  double PR = (gamma-1.0) * (uR_p[4] - 0.5*rhoR*vsqR);
  double HL = 0.5*vsqL + gamma*PL/((gamma-1.0)*rhoL);
  double HR = 0.5*vsqR + gamma*PR/((gamma-1.0)*rhoR);

  /* Roe-averaged quantities */
  double vx = (tL*vxL + tR*vxR) / tLpR;
  double vy = (tL*vyL + tR*vyR) / tLpR;
  double vz = (tL*vzL + tR*vzR) / tLpR;
  double H = (tL*HL + tR*HR) / tLpR;
  double vsq = vx*vx + vy*vy + vz*vz;
  double a2 = (gamma-1.0) * (H - 0.5*vsq);
  double a = sqrt(a2);

  /* Normal velocity for eigenvalues */
  double vn = (dir == 0) ? vx : ((dir == 1) ? vy : vz);

  /* ===== Eigenvalues for all 12 variables ===== */
  double eigL[12], eigC[12], eigR[12];

  /* Left state eigenvalues */
  double cL = sqrt(gamma * PL / rhoL);
  double vnL = (dir == 0) ? vxL : ((dir == 1) ? vyL : vzL);
  eigL[0] = vnL;
  eigL[1] = (dir == 0) ? vnL - cL : vnL;
  eigL[2] = (dir == 1) ? vnL - cL : vnL;
  eigL[3] = (dir == 2) ? vnL - cL : vnL;
  eigL[4] = vnL + cL;
  #pragma unroll
  for (int k = 5; k < 12; k++) eigL[k] = vnL;

  /* Right state eigenvalues */
  double cR = sqrt(gamma * PR / rhoR);
  double vnR = (dir == 0) ? vxR : ((dir == 1) ? vyR : vzR);
  eigR[0] = vnR;
  eigR[1] = (dir == 0) ? vnR - cR : vnR;
  eigR[2] = (dir == 1) ? vnR - cR : vnR;
  eigR[3] = (dir == 2) ? vnR - cR : vnR;
  eigR[4] = vnR + cR;
  #pragma unroll
  for (int k = 5; k < 12; k++) eigR[k] = vnR;

  /* Roe-averaged eigenvalues */
  eigC[0] = vn;
  eigC[1] = (dir == 0) ? vn - a : vn;
  eigC[2] = (dir == 1) ? vn - a : vn;
  eigC[3] = (dir == 2) ? vn - a : vn;
  eigC[4] = vn + a;
  #pragma unroll
  for (int k = 5; k < 12; k++) eigC[k] = vn;

  /* ===== Build 5x5 eigenvector matrices (registers) ===== */
  double L[25], R[25];

  double gm1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double a2inv = 1.0 / a2;
  double twoA2inv = 0.5 * a2inv;
  double h0 = a2/gm1 + ek;

  /* Initialize to zero */
  #pragma unroll
  for (int i = 0; i < 25; i++) { L[i] = 0.0; R[i] = 0.0; }

  /* Build L and R based on direction */
  if (dir == 0) { /* X-direction */
    L[5*1+0] = (gm1*ek + a*vx) * twoA2inv;
    L[5*1+1] = (-gm1*vx - a) * twoA2inv;
    L[5*1+2] = (-gm1*vy) * twoA2inv;
    L[5*1+3] = (-gm1*vz) * twoA2inv;
    L[5*1+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv;
    L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv;
    L[5*0+3] = gm1*vz * a2inv;
    L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vx) * twoA2inv;
    L[5*4+1] = (-gm1*vx + a) * twoA2inv;
    L[5*4+2] = (-gm1*vy) * twoA2inv;
    L[5*4+3] = (-gm1*vz) * twoA2inv;
    L[5*4+4] = gm1 * twoA2inv;
    L[5*2+0] = vy; L[5*2+2] = -1.0;
    L[5*3+0] = -vz; L[5*3+3] = 1.0;

    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[0*5+1] = 1.0; R[1*5+1] = vx-a; R[2*5+1] = vy; R[3*5+1] = vz; R[4*5+1] = h0-a*vx;
    R[2*5+2] = -1.0; R[4*5+2] = -vy;
    R[3*5+3] = 1.0; R[4*5+3] = vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx+a; R[2*5+4] = vy; R[3*5+4] = vz; R[4*5+4] = h0+a*vx;
  } else if (dir == 1) { /* Y-direction */
    L[5*2+0] = (gm1*ek + a*vy) * twoA2inv;
    L[5*2+1] = (-gm1*vx) * twoA2inv;
    L[5*2+2] = (-gm1*vy - a) * twoA2inv;
    L[5*2+3] = (-gm1*vz) * twoA2inv;
    L[5*2+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv;
    L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv;
    L[5*0+3] = gm1*vz * a2inv;
    L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vy) * twoA2inv;
    L[5*4+1] = (-gm1*vx) * twoA2inv;
    L[5*4+2] = (-gm1*vy + a) * twoA2inv;
    L[5*4+3] = (-gm1*vz) * twoA2inv;
    L[5*4+4] = gm1 * twoA2inv;
    L[5*1+0] = -vx; L[5*1+1] = 1.0;
    L[5*3+0] = vz; L[5*3+3] = -1.0;

    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = 1.0; R[4*5+1] = vx;
    R[0*5+2] = 1.0; R[1*5+2] = vx; R[2*5+2] = vy-a; R[3*5+2] = vz; R[4*5+2] = h0-a*vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy+a; R[3*5+4] = vz; R[4*5+4] = h0+a*vy;
  } else { /* Z-direction */
    L[5*3+0] = (gm1*ek + a*vz) * twoA2inv;
    L[5*3+1] = (-gm1*vx) * twoA2inv;
    L[5*3+2] = (-gm1*vy) * twoA2inv;
    L[5*3+3] = (-gm1*vz - a) * twoA2inv;
    L[5*3+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv;
    L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv;
    L[5*0+3] = gm1*vz * a2inv;
    L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vz) * twoA2inv;
    L[5*4+1] = (-gm1*vx) * twoA2inv;
    L[5*4+2] = (-gm1*vy) * twoA2inv;
    L[5*4+3] = (-gm1*vz + a) * twoA2inv;
    L[5*4+4] = gm1 * twoA2inv;
    L[5*1+0] = vx; L[5*1+1] = -1.0;
    L[5*2+0] = -vy; L[5*2+2] = 1.0;

    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = -1.0; R[4*5+1] = -vx;
    R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[0*5+3] = 1.0; R[1*5+3] = vx; R[2*5+3] = vy; R[3*5+3] = vz-a; R[4*5+3] = h0-a*vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy; R[3*5+4] = vz+a; R[4*5+4] = h0+a*vz;
  }

  /* ===== Transform base 5 variables to characteristic space ===== */
  double ucL[5], ucR[5], fcL[5], fcR[5], fc[5];

  #pragma unroll
  for (int i = 0; i < 5; i++) {
    ucL[i] = L[i*5+0]*uL_p[0] + L[i*5+1]*uL_p[1] + L[i*5+2]*uL_p[2] + L[i*5+3]*uL_p[3] + L[i*5+4]*uL_p[4];
    ucR[i] = L[i*5+0]*uR_p[0] + L[i*5+1]*uR_p[1] + L[i*5+2]*uR_p[2] + L[i*5+3]*uR_p[3] + L[i*5+4]*uR_p[4];
    fcL[i] = L[i*5+0]*fL_p[0] + L[i*5+1]*fL_p[1] + L[i*5+2]*fL_p[2] + L[i*5+3]*fL_p[3] + L[i*5+4]*fL_p[4];
    fcR[i] = L[i*5+0]*fR_p[0] + L[i*5+1]*fR_p[1] + L[i*5+2]*fR_p[2] + L[i*5+3]*fR_p[3] + L[i*5+4]*fR_p[4];
  }

  /* ===== Compute characteristic fluxes for base 5 variables ===== */
  #pragma unroll
  for (int k = 0; k < 5; k++) {
    if ((eigL[k] > 0) && (eigC[k] > 0) && (eigR[k] > 0)) {
      fc[k] = fcL[k];
    } else if ((eigL[k] < 0) && (eigC[k] < 0) && (eigR[k] < 0)) {
      fc[k] = fcR[k];
    } else {
      double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
      fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
    }
  }

  /* ===== Transform base 5 variables back to physical space ===== */
  #pragma unroll
  for (int i = 0; i < 5; i++) {
    fI_p[i] = R[i*5+0]*fc[0] + R[i*5+1]*fc[1] + R[i*5+2]*fc[2] + R[i*5+3]*fc[3] + R[i*5+4]*fc[4];
  }

  /* ===== Simple upwinding for passive scalars (vars 5-11) ===== */
  /* These use identity eigenvectors, so characteristic = physical */
  #pragma unroll
  for (int k = 5; k < 12; k++) {
    if ((eigL[k] > 0) && (eigC[k] > 0) && (eigR[k] > 0)) {
      fI_p[k] = fL_p[k];
    } else if ((eigL[k] < 0) && (eigC[k] < 0) && (eigR[k] < 0)) {
      fI_p[k] = fR_p[k];
    } else {
      double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
      fI_p[k] = 0.5 * (fL_p[k] + fR_p[k] + alpha * (uL_p[k] - uR_p[k]));
    }
  }
}

/* Kernel: RF (Roe-Fixed) upwinding for NavierStokes3D - general fallback */
GPU_KERNEL void gpu_ns3d_upwind_rf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  /* Unrolled for 3D */
  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    /* Decompose idx - unrolled for 3D */
    int indexI[NS3D_NDIMS];
    int temp = idx;
    indexI[2] = temp % bounds_inter[2]; temp /= bounds_inter[2];
    indexI[1] = temp % bounds_inter[1]; temp /= bounds_inter[1];
    indexI[0] = temp;
    
    /* Compute 1D interface index - unrolled */
    int p = indexI[2] + bounds_inter[2] * (indexI[1] + bounds_inter[1] * indexI[0]);
    
    /* Compute cell indices - unrolled */
    int indexL[NS3D_NDIMS] = {indexI[0], indexI[1], indexI[2]};
    indexL[dir]--;

    /* RF scheme - use dynamic workspace */
    /* Workspace layout: [uavg, fcL, fcR, ucL, ucR, fc, eigL, eigC, eigR, L, R] */
    /* Total: 9*nvars + 2*nvars*nvars per thread */
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t workspace_per_thread = 9 * nvars + 2 * nvars * nvars;
    double *thread_workspace = workspace + threadId * workspace_per_thread;
    
    double *uavg = thread_workspace;
    double *fcL = uavg + nvars;
    double *fcR = fcL + nvars;
    double *ucL = fcR + nvars;
    double *ucR = ucL + nvars;
    double *fc = ucR + nvars;
    double *eigL = fc + nvars;
    double *eigC = eigL + nvars;
    double *eigR = eigC + nvars;
    double *L = eigR + nvars;
    double *R = L + nvars * nvars;
    
    gpu_ns3d_roe_average(uavg, uL + p*nvars, uR + p*nvars, nvars, gamma);
    gpu_ns3d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns3d_right_eigenvectors(uavg, R, gamma, nvars, dir);
    
    gpu_matvecmult(nvars, ucL, L, uL + p*nvars);
    gpu_matvecmult(nvars, ucR, L, uR + p*nvars);
    gpu_matvecmult(nvars, fcL, L, fL + p*nvars);
    gpu_matvecmult(nvars, fcR, L, fR + p*nvars);
    
    /* Use optimized eigenvalue extraction */
    gpu_ns3d_eigenvalues_diag(uL + p*nvars, eigL, gamma, nvars, dir);
    gpu_ns3d_eigenvalues_diag(uR + p*nvars, eigR, gamma, nvars, dir);
    gpu_ns3d_eigenvalues_diag(uavg, eigC, gamma, nvars, dir);
    
    for (int k = 0; k < nvars; k++) {
      if ((eigL[k] > 0) && (eigC[k] > 0) && (eigR[k] > 0)) {
        fc[k] = fcL[k];
      } else if ((eigL[k] < 0) && (eigC[k] < 0) && (eigR[k] < 0)) {
        fc[k] = fcR[k];
      } else {
        double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
        fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
      }
    }
    
    gpu_matvecmult(nvars, fI + p*nvars, R, fc);
  }
}

/* Kernel: LLF upwinding optimized for nvars=5
 * All computation in registers, no global memory workspace needed.
 */
GPU_KERNEL void gpu_ns3d_upwind_llf_kernel_nvars5(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  const int nvars = 5;

  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_interfaces) return;

  /* Decompose idx - unrolled for 3D */
  int temp = idx;
  const int i2 = temp % bounds_inter[2]; temp /= bounds_inter[2];
  const int i1 = temp % bounds_inter[1]; temp /= bounds_inter[1];
  const int i0 = temp;

  /* Compute 1D interface index */
  int p = i2 + bounds_inter[2] * (i1 + bounds_inter[1] * i0);

  /* Register-based storage */
  double uavg[5], ucL[5], ucR[5], fcL[5], fcR[5], fc[5];
  double eigL[5], eigC[5], eigR[5];
  double L[25], R[25];

  /* Pointers for readability */
  const double *uL_p = uL + p * nvars;
  const double *uR_p = uR + p * nvars;
  const double *fL_p = fL + p * nvars;
  const double *fR_p = fR + p * nvars;
  double *fI_p = fI + p * nvars;

  /* Roe average */
  gpu_ns3d_roe_average(uavg, uL_p, uR_p, nvars, gamma);

  /* Eigenvectors */
  gpu_ns3d_left_eigenvectors(uavg, L, gamma, nvars, dir);
  gpu_ns3d_right_eigenvectors(uavg, R, gamma, nvars, dir);

  /* Transform to characteristic space */
  gpu_matvecmult_5(ucL, L, uL_p);
  gpu_matvecmult_5(ucR, L, uR_p);
  gpu_matvecmult_5(fcL, L, fL_p);
  gpu_matvecmult_5(fcR, L, fR_p);

  /* Eigenvalues */
  gpu_ns3d_eigenvalues_diag(uL_p, eigL, gamma, nvars, dir);
  gpu_ns3d_eigenvalues_diag(uR_p, eigR, gamma, nvars, dir);
  gpu_ns3d_eigenvalues_diag(uavg, eigC, gamma, nvars, dir);

  /* LLF characteristic fluxes */
  #pragma unroll
  for (int k = 0; k < 5; k++) {
    double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
    fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
  }

  /* Transform back to physical space */
  gpu_matvecmult_5(fI_p, R, fc);
}

/* Kernel: LLF upwinding optimized for nvars=12
 * Key optimization: Only compute 5x5 eigenvector block for base flow variables.
 * Passive scalars (vars 5-11) use identity eigenvectors (simple LLF).
 * All computation in registers, no global memory workspace needed.
 */
GPU_KERNEL void gpu_ns3d_upwind_llf_kernel_nvars12(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  const int nvars = 12;
  /* Note: Only 5 base flow variables need characteristic decomposition;
     passive scalars (vars 5-11) use simple LLF with identity eigenvectors */

  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_interfaces) return;

  /* Decompose idx - unrolled for 3D */
  int temp = idx;
  const int i2 = temp % bounds_inter[2]; temp /= bounds_inter[2];
  const int i1 = temp % bounds_inter[1]; temp /= bounds_inter[1];
  const int i0 = temp;

  /* Compute 1D interface index */
  int p = i2 + bounds_inter[2] * (i1 + bounds_inter[1] * i0);

  /* Pointers for readability */
  const double *uL_p = uL + p * nvars;
  const double *uR_p = uR + p * nvars;
  const double *fL_p = fL + p * nvars;
  const double *fR_p = fR + p * nvars;
  double *fI_p = fI + p * nvars;

  /* ===== Compute Roe average for base 5 variables ===== */
  double rhoL = uL_p[0], rhoR = uR_p[0];
  if (rhoL <= 0.0 || rhoR <= 0.0) {
    for (int k = 0; k < nvars; k++) fI_p[k] = 0.5 * (fL_p[k] + fR_p[k]);
    return;
  }

  double sqrtRhoL = sqrt(rhoL), sqrtRhoR = sqrt(rhoR);
  double denom = sqrtRhoL + sqrtRhoR;
  double vxL = uL_p[1]/rhoL, vyL = uL_p[2]/rhoL, vzL = uL_p[3]/rhoL;
  double vxR = uR_p[1]/rhoR, vyR = uR_p[2]/rhoR, vzR = uR_p[3]/rhoR;

  double vx = (sqrtRhoL*vxL + sqrtRhoR*vxR) / denom;
  double vy = (sqrtRhoL*vyL + sqrtRhoR*vyR) / denom;
  double vz = (sqrtRhoL*vzL + sqrtRhoR*vzR) / denom;

  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  double PL = (gamma-1.0) * (uL_p[4] - 0.5*rhoL*vsqL);
  double PR = (gamma-1.0) * (uR_p[4] - 0.5*rhoR*vsqR);
  if (PL <= 0.0 || PR <= 0.0) {
    for (int k = 0; k < nvars; k++) fI_p[k] = 0.5 * (fL_p[k] + fR_p[k]);
    return;
  }

  double hL = (uL_p[4] + PL) / rhoL;
  double hR = (uR_p[4] + PR) / rhoR;
  double h0 = (sqrtRhoL*hL + sqrtRhoR*hR) / denom;

  double vsq = vx*vx + vy*vy + vz*vz;
  double a2 = (gamma-1.0) * (h0 - 0.5*vsq);
  if (a2 <= 0.0) {
    for (int k = 0; k < nvars; k++) fI_p[k] = 0.5 * (fL_p[k] + fR_p[k]);
    return;
  }
  double a = sqrt(a2);

  /* ===== Compute eigenvalues for all 12 variables ===== */
  double eigL[12], eigC[12], eigR[12];
  double vn, vnL, vnR;

  if (dir == _XDIR_) {
    vnL = vxL; vnR = vxR; vn = vx;
  } else if (dir == _YDIR_) {
    vnL = vyL; vnR = vyR; vn = vy;
  } else {
    vnL = vzL; vnR = vzR; vn = vz;
  }

  double cL = sqrt(gamma * PL / rhoL);
  double cR = sqrt(gamma * PR / rhoR);

  /* Eigenvalues for L, C, R states */
  eigL[0] = vnL; eigL[1] = vnL; eigL[2] = vnL; eigL[3] = vnL - cL; eigL[4] = vnL + cL;
  eigC[0] = vn;  eigC[1] = vn;  eigC[2] = vn;  eigC[3] = vn - a;   eigC[4] = vn + a;
  eigR[0] = vnR; eigR[1] = vnR; eigR[2] = vnR; eigR[3] = vnR - cR; eigR[4] = vnR + cR;

  /* Passive scalars have same eigenvalue as density (convected with flow) */
  for (int k = 5; k < 12; k++) {
    eigL[k] = vnL; eigC[k] = vn; eigR[k] = vnR;
  }

  /* ===== Build 5x5 eigenvector matrices inline ===== */
  double L[25] = {0}, R[25] = {0};
  double gm1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double a2inv = 1.0 / a2;
  double twoA2inv = 0.5 * a2inv;

  if (dir == _XDIR_) {
    L[5*1+0] = (gm1*ek + a*vx) * twoA2inv;
    L[5*1+1] = (-gm1*vx - a) * twoA2inv;
    L[5*1+2] = (-gm1*vy) * twoA2inv;
    L[5*1+3] = (-gm1*vz) * twoA2inv;
    L[5*1+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv;
    L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv;
    L[5*0+3] = gm1*vz * a2inv;
    L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vx) * twoA2inv;
    L[5*4+1] = (-gm1*vx + a) * twoA2inv;
    L[5*4+2] = (-gm1*vy) * twoA2inv;
    L[5*4+3] = (-gm1*vz) * twoA2inv;
    L[5*4+4] = gm1 * twoA2inv;
    L[5*2+0] = -vy; L[5*2+2] = 1.0;
    L[5*3+0] = vz; L[5*3+3] = -1.0;

    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[0*5+1] = 1.0; R[1*5+1] = vx-a; R[2*5+1] = vy; R[3*5+1] = vz; R[4*5+1] = h0-a*vx;
    R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx+a; R[2*5+4] = vy; R[3*5+4] = vz; R[4*5+4] = h0+a*vx;
  } else if (dir == _YDIR_) {
    L[5*2+0] = (gm1*ek + a*vy) * twoA2inv;
    L[5*2+1] = (-gm1*vx) * twoA2inv;
    L[5*2+2] = (-gm1*vy - a) * twoA2inv;
    L[5*2+3] = (-gm1*vz) * twoA2inv;
    L[5*2+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv;
    L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv;
    L[5*0+3] = gm1*vz * a2inv;
    L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vy) * twoA2inv;
    L[5*4+1] = (-gm1*vx) * twoA2inv;
    L[5*4+2] = (-gm1*vy + a) * twoA2inv;
    L[5*4+3] = (-gm1*vz) * twoA2inv;
    L[5*4+4] = gm1 * twoA2inv;
    L[5*1+0] = -vx; L[5*1+1] = 1.0;
    L[5*3+0] = vz; L[5*3+3] = -1.0;

    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = 1.0; R[4*5+1] = vx;
    R[0*5+2] = 1.0; R[1*5+2] = vx; R[2*5+2] = vy-a; R[3*5+2] = vz; R[4*5+2] = h0-a*vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy+a; R[3*5+4] = vz; R[4*5+4] = h0+a*vy;
  } else { /* Z-direction */
    L[5*3+0] = (gm1*ek + a*vz) * twoA2inv;
    L[5*3+1] = (-gm1*vx) * twoA2inv;
    L[5*3+2] = (-gm1*vy) * twoA2inv;
    L[5*3+3] = (-gm1*vz - a) * twoA2inv;
    L[5*3+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv;
    L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv;
    L[5*0+3] = gm1*vz * a2inv;
    L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vz) * twoA2inv;
    L[5*4+1] = (-gm1*vx) * twoA2inv;
    L[5*4+2] = (-gm1*vy) * twoA2inv;
    L[5*4+3] = (-gm1*vz + a) * twoA2inv;
    L[5*4+4] = gm1 * twoA2inv;
    L[5*1+0] = vx; L[5*1+1] = -1.0;
    L[5*2+0] = -vy; L[5*2+2] = 1.0;

    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = -1.0; R[4*5+1] = -vx;
    R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[0*5+3] = 1.0; R[1*5+3] = vx; R[2*5+3] = vy; R[3*5+3] = vz-a; R[4*5+3] = h0-a*vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy; R[3*5+4] = vz+a; R[4*5+4] = h0+a*vz;
  }

  /* ===== Transform base 5 variables to characteristic space ===== */
  double ucL[5], ucR[5], fcL[5], fcR[5], fc[5];

  #pragma unroll
  for (int i = 0; i < 5; i++) {
    ucL[i] = L[i*5+0]*uL_p[0] + L[i*5+1]*uL_p[1] + L[i*5+2]*uL_p[2] + L[i*5+3]*uL_p[3] + L[i*5+4]*uL_p[4];
    ucR[i] = L[i*5+0]*uR_p[0] + L[i*5+1]*uR_p[1] + L[i*5+2]*uR_p[2] + L[i*5+3]*uR_p[3] + L[i*5+4]*uR_p[4];
    fcL[i] = L[i*5+0]*fL_p[0] + L[i*5+1]*fL_p[1] + L[i*5+2]*fL_p[2] + L[i*5+3]*fL_p[3] + L[i*5+4]*fL_p[4];
    fcR[i] = L[i*5+0]*fR_p[0] + L[i*5+1]*fR_p[1] + L[i*5+2]*fR_p[2] + L[i*5+3]*fR_p[3] + L[i*5+4]*fR_p[4];
  }

  /* ===== LLF characteristic fluxes for base 5 variables ===== */
  #pragma unroll
  for (int k = 0; k < 5; k++) {
    double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
    fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
  }

  /* ===== Transform base 5 variables back to physical space ===== */
  #pragma unroll
  for (int i = 0; i < 5; i++) {
    fI_p[i] = R[i*5+0]*fc[0] + R[i*5+1]*fc[1] + R[i*5+2]*fc[2] + R[i*5+3]*fc[3] + R[i*5+4]*fc[4];
  }

  /* ===== Simple LLF for passive scalars (vars 5-11) ===== */
  /* These use identity eigenvectors, so characteristic = physical */
  #pragma unroll
  for (int k = 5; k < 12; k++) {
    double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
    fI_p[k] = 0.5 * (fL_p[k] + fR_p[k] + alpha * (uL_p[k] - uR_p[k]));
  }
}

/* Kernel: LLF (Local Lax-Friedrich) upwinding for NavierStokes3D - general fallback */
GPU_KERNEL void gpu_ns3d_upwind_llf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    int indexI[3];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      indexI[i] = temp % bounds_inter[i];
      temp /= bounds_inter[i];
    }
    
    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }
    
    int indexL[3], indexR[3];
    for (int i = 0; i < ndims; i++) {
      indexL[i] = indexI[i];
      indexR[i] = indexI[i];
    }
    indexL[dir]--;
    
    int pL = 0, pR = 0;
    for (int i = 0; i < ndims; i++) {
      pL += (indexL[i] + ghosts) * stride_with_ghosts[i];
      pR += (indexR[i] + ghosts) * stride_with_ghosts[i];
    }
    
    /* LLF scheme - use dynamic workspace */
    /* Workspace layout: [uavg, fcL, fcR, ucL, ucR, fc, eigL, eigC, eigR, D, L, R] */
    /* Total: 9*nvars + 3*nvars*nvars per thread */
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t workspace_per_thread = 9 * nvars + 3 * nvars * nvars;
    double *thread_workspace = workspace + threadId * workspace_per_thread;
    
    double *uavg = thread_workspace;
    double *fcL = uavg + nvars;
    double *fcR = fcL + nvars;
    double *ucL = fcR + nvars;
    double *ucR = ucL + nvars;
    double *fc = ucR + nvars;
    double *eigL = fc + nvars;
    double *eigC = eigL + nvars;
    double *eigR = eigC + nvars;
    double *D = eigR + nvars;
    double *L = D + nvars * nvars;
    double *R = L + nvars * nvars;
    
    gpu_ns3d_roe_average(uavg, uL + p*nvars, uR + p*nvars, nvars, gamma);
    gpu_ns3d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns3d_right_eigenvectors(uavg, R, gamma, nvars, dir);
    
    gpu_matvecmult(nvars, ucL, L, uL + p*nvars);
    gpu_matvecmult(nvars, ucR, L, uR + p*nvars);
    gpu_matvecmult(nvars, fcL, L, fL + p*nvars);
    gpu_matvecmult(nvars, fcR, L, fR + p*nvars);
    
    gpu_ns3d_eigenvalues(uL + p*nvars, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigL[k] = D[k*nvars+k];
    gpu_ns3d_eigenvalues(uR + p*nvars, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigR[k] = D[k*nvars+k];
    gpu_ns3d_eigenvalues(uavg, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigC[k] = D[k*nvars+k];
    
    for (int k = 0; k < nvars; k++) {
      double alpha = gpu_max3(gpu_absolute(eigL[k]), gpu_absolute(eigC[k]), gpu_absolute(eigR[k]));
      fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
    }
    
    gpu_matvecmult(nvars, fI + p*nvars, R, fc);
  }
}

/* Kernel: Rusanov upwinding for NavierStokes3D */
GPU_KERNEL void gpu_ns3d_upwind_rusanov_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, double *workspace
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    int indexI[3];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      indexI[i] = temp % bounds_inter[i];
      temp /= bounds_inter[i];
    }
    
    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }
    
    int indexL[3], indexR[3];
    for (int i = 0; i < ndims; i++) {
      indexL[i] = indexI[i];
      indexR[i] = indexI[i];
    }
    indexL[dir]--;
    
    int pL = 0, pR = 0;
    for (int i = 0; i < ndims; i++) {
      pL += (indexL[i] + ghosts) * stride_with_ghosts[i];
      pR += (indexR[i] + ghosts) * stride_with_ghosts[i];
    }
    
    /* Rusanov scheme - use dynamic workspace */
    /* Workspace layout: [uavg, udiff] */
    /* Total: 2*nvars per thread */
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t workspace_per_thread = 2 * nvars;
    double *thread_workspace = workspace + threadId * workspace_per_thread;
    
    double *uavg = thread_workspace;
    double *udiff = uavg + nvars;
    
    for (int k = 0; k < nvars; k++) {
      udiff[k] = 0.5 * (uR[p*nvars+k] - uL[p*nvars+k]);
    }
    
    gpu_ns3d_roe_average(uavg, u + pL*nvars, u + pR*nvars, nvars, gamma);
    
    double rho, vel[3], e, P;
    rho = u[pL*nvars + 0];
    vel[0] = (rho == 0) ? 0 : u[pL*nvars + 1] / rho;
    vel[1] = (rho == 0) ? 0 : u[pL*nvars + 2] / rho;
    vel[2] = (rho == 0) ? 0 : u[pL*nvars + 3] / rho;
    e = u[pL*nvars + 4];
    double vsq = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
    P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
    double cL = sqrt(gamma * P / rho);
    double alphaL = cL + gpu_absolute(vel[dir]);
    double betaL = gpu_absolute(vel[dir]);
    
    rho = u[pR*nvars + 0];
    vel[0] = (rho == 0) ? 0 : u[pR*nvars + 1] / rho;
    vel[1] = (rho == 0) ? 0 : u[pR*nvars + 2] / rho;
    vel[2] = (rho == 0) ? 0 : u[pR*nvars + 3] / rho;
    e = u[pR*nvars + 4];
    vsq = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
    P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
    double cR = sqrt(gamma * P / rho);
    double alphaR = cR + gpu_absolute(vel[dir]);
    double betaR = gpu_absolute(vel[dir]);
    
    rho = uavg[0];
    vel[0] = (rho == 0) ? 0 : uavg[1] / rho;
    vel[1] = (rho == 0) ? 0 : uavg[2] / rho;
    vel[2] = (rho == 0) ? 0 : uavg[3] / rho;
    e = uavg[4];
    vsq = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2];
    P = (gamma - 1.0) * (e - 0.5 * rho * vsq);
    double cavg = sqrt(gamma * P / rho);
    double alphaavg = cavg + gpu_absolute(vel[dir]);
    double betaavg = gpu_absolute(vel[dir]);
    
    double alpha = gpu_max3(alphaL, alphaR, alphaavg);
    double beta = gpu_max3(betaL, betaR, betaavg);
    
    for (int k = 0; k < nvars; k++) {
      fI[p*nvars+k] = 0.5 * (fL[p*nvars+k] + fR[p*nvars+k]) - alpha * udiff[k];
    }
  }
}

