/*! @file gpu_upwind.cu
    @brief GPU kernels for NavierStokes3D upwinding
*/

#include <gpu.h>
#include <physicalmodels/gpu_ns3d_helpers.h>
#include <math.h>

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
  /* Compute total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    /* Decompose idx into multi-dimensional interface index */
    int indexI[3]; /* Support up to 3D */
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      indexI[i] = temp % bounds_inter[i];
      temp /= bounds_inter[i];
    }
    
    /* Compute 1D index for interface point (no ghosts) */
    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }
    
    /* Compute cell indices for left and right cells */
    int indexL[3], indexR[3];
    for (int i = 0; i < ndims; i++) {
      indexL[i] = indexI[i];
      indexR[i] = indexI[i];
    }
    indexL[dir]--;
    
    /* Compute 1D indices for cell-centered arrays (with ghosts) */
    int pL = 0, pR = 0;
    for (int i = 0; i < ndims; i++) {
      pL += (indexL[i] + ghosts) * stride_with_ghosts[i];
      pR += (indexR[i] + ghosts) * stride_with_ghosts[i];
    }
    
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

/* Kernel: RF (Roe-Fixed) upwinding for NavierStokes3D */
GPU_KERNEL void gpu_ns3d_upwind_rf_kernel(
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
    
    /* RF scheme - use dynamic workspace */
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

/* Kernel: LLF (Local Lax-Friedrich) upwinding for NavierStokes3D */
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

