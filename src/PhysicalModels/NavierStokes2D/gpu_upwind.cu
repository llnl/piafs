/*! @file gpu_upwind.cu
    @brief GPU kernels for NavierStokes2D upwinding
*/

#include <gpu.h>
#include <physicalmodels/gpu_ns2d_helpers.h>
#include <math.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Helper function: Compute absolute value */
__device__ double gpu_ns2d_absolute(double x) {
  return (x < 0.0) ? -x : x;
}

/* Helper function: Matrix multiply */
__device__ void gpu_ns2d_matmult(int n, double *C, const double *A, const double *B) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i*n + j] = 0.0;
      for (int k = 0; k < n; k++) {
        C[i*n + j] += A[i*n + k] * B[k*n + j];
      }
    }
  }
}

/* Helper function: Matrix-vector multiply */
__device__ void gpu_ns2d_matvecmult(int n, double *y, const double *A, const double *x) {
  for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = 0; j < n; j++) {
      y[i] += A[i*n + j] * x[j];
    }
  }
}

/* Helper function: max3 */
__device__ double gpu_ns2d_max3(double a, double b, double c) {
  double ab = (a > b) ? a : b;
  return (ab > c) ? ab : c;
}

/* Helper function: Compute eigenvalues for NavierStokes2D */
__device__ void gpu_ns2d_eigenvalues(const double *u, double *D, double gamma, int nvars, int dir) {
  double rho = u[0];
  if (rho == 0.0) {
    for (int i = 0; i < nvars*nvars; i++) D[i] = 0.0;
    return;
  }
  double vx = u[1] / rho;
  double vy = u[2] / rho;
  double e = u[3];
  double vsq = vx*vx + vy*vy;
  double P = (e - 0.5*rho*vsq) * (gamma - 1.0);
  if (P <= 0.0 || rho <= 0.0) {
    for (int i = 0; i < nvars*nvars; i++) D[i] = 0.0;
    return;
  }
  double c = sqrt(gamma * P / rho);

  double vn;
  if (dir == _XDIR_) vn = vx;
  else if (dir == _YDIR_) vn = vy;
  else vn = 0.0;

  /* Initialize D to zero */
  for (int i = 0; i < nvars*nvars; i++) D[i] = 0.0;

  D[0*nvars+0] = vn - c;
  if (dir == _XDIR_) {
    D[1*nvars+1] = vn + c;
    D[2*nvars+2] = vn;
  } else if (dir == _YDIR_) {
    D[1*nvars+1] = vn;
    D[2*nvars+2] = vn + c;
  }
  D[3*nvars+3] = vn;
  for (int m_i = _NS2D_NVARS_; m_i < nvars; m_i++) {
    D[m_i*nvars+m_i] = vn;
  }
}

/* Kernel: Roe upwinding for NavierStokes2D
   Each thread handles one interface point
*/
GPU_KERNEL void gpu_ns2d_upwind_roe_kernel(
  double *fI,              /* output: upwind interface flux */
  const double *fL,        /* input: left-biased flux */
  const double *fR,        /* input: right-biased flux */
  const double *uL,        /* input: left-biased solution */
  const double *uR,        /* input: right-biased solution */
  const double *u,         /* input: cell-centered solution */
  int nvars,               /* number of variables */
  int ndims,               /* number of dimensions */
  const int *dim,          /* dimension sizes (without ghosts) */
  const int *stride_with_ghosts, /* stride array */
  const int *bounds_inter, /* bounds for interface array */
  int ghosts,              /* number of ghost points */
  int dir,                 /* direction */
  double gamma             /* gamma parameter */
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
    int indexI[2];
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
    int indexL[2], indexR[2];
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

    /* Roe's upwinding scheme */
    const int max_nvars = 10;
    double udiff[10], uavg[10], udiss[10];
    if (nvars > max_nvars) return; /* Safety check */

    for (int k = 0; k < nvars; k++) {
      udiff[k] = 0.5 * (uR[p*nvars+k] - uL[p*nvars+k]);
    }

    /* Compute Roe average */
    gpu_ns2d_roe_average(uavg, u + pL*nvars, u + pR*nvars, nvars, gamma);

    /* Compute eigenvalues, left and right eigenvectors */
    double D[100], L[100], R[100], DL[100], modA[100];
    if (nvars * nvars > 100) return; /* Safety check */

    gpu_ns2d_eigenvalues(uavg, D, gamma, nvars, dir);
    gpu_ns2d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns2d_right_eigenvectors(uavg, R, gamma, nvars, dir);

    /* Harten's Entropy Fix */
    double delta = 0.000001, delta2 = delta*delta;
    for (int k = 0; k < nvars; k++) {
      double absD = gpu_ns2d_absolute(D[k*nvars+k]);
      D[k*nvars+k] = (absD < delta) ? (D[k*nvars+k]*D[k*nvars+k]+delta2)/(2*delta) : absD;
    }

    /* Compute modA = R * |D| * L */
    gpu_ns2d_matmult(nvars, DL, D, L);
    gpu_ns2d_matmult(nvars, modA, R, DL);

    /* Compute dissipation: modA * udiff */
    gpu_ns2d_matvecmult(nvars, udiss, modA, udiff);

    /* Compute upwind flux */
    for (int k = 0; k < nvars; k++) {
      fI[p*nvars+k] = 0.5 * (fL[p*nvars+k] + fR[p*nvars+k]) - udiss[k];
    }
  }
}

/* Kernel: RF (Roe-Fixed) upwinding for NavierStokes2D */
GPU_KERNEL void gpu_ns2d_upwind_rf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    int indexI[2];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      indexI[i] = temp % bounds_inter[i];
      temp /= bounds_inter[i];
    }

    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }

    int indexL[2], indexR[2];
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

    const int max_nvars = 10;
    double uavg[10], fcL[10], fcR[10], ucL[10], ucR[10], fc[10];
    double L[100], R[100], D[100];
    if (nvars > max_nvars || nvars * nvars > 100) return;

    gpu_ns2d_roe_average(uavg, uL + p*nvars, uR + p*nvars, nvars, gamma);
    gpu_ns2d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns2d_right_eigenvectors(uavg, R, gamma, nvars, dir);

    gpu_ns2d_matvecmult(nvars, ucL, L, uL + p*nvars);
    gpu_ns2d_matvecmult(nvars, ucR, L, uR + p*nvars);
    gpu_ns2d_matvecmult(nvars, fcL, L, fL + p*nvars);
    gpu_ns2d_matvecmult(nvars, fcR, L, fR + p*nvars);

    double eigL[10], eigC[10], eigR[10];
    gpu_ns2d_eigenvalues(uL + p*nvars, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigL[k] = D[k*nvars+k];
    gpu_ns2d_eigenvalues(uR + p*nvars, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigR[k] = D[k*nvars+k];
    gpu_ns2d_eigenvalues(uavg, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigC[k] = D[k*nvars+k];

    for (int k = 0; k < nvars; k++) {
      if ((eigL[k] > 0) && (eigC[k] > 0) && (eigR[k] > 0)) {
        fc[k] = fcL[k];
      } else if ((eigL[k] < 0) && (eigC[k] < 0) && (eigR[k] < 0)) {
        fc[k] = fcR[k];
      } else {
        double alpha = gpu_ns2d_max3(gpu_ns2d_absolute(eigL[k]), gpu_ns2d_absolute(eigC[k]), gpu_ns2d_absolute(eigR[k]));
        fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
      }
    }

    gpu_ns2d_matvecmult(nvars, fI + p*nvars, R, fc);
  }
}

/* Kernel: LLF (Local Lax-Friedrich) upwinding for NavierStokes2D */
GPU_KERNEL void gpu_ns2d_upwind_llf_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    int indexI[2];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      indexI[i] = temp % bounds_inter[i];
      temp /= bounds_inter[i];
    }

    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }

    int indexL[2], indexR[2];
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

    const int max_nvars = 10;
    double uavg[10], fcL[10], fcR[10], ucL[10], ucR[10], fc[10];
    double L[100], R[100], D[100];
    if (nvars > max_nvars || nvars * nvars > 100) return;

    gpu_ns2d_roe_average(uavg, uL + p*nvars, uR + p*nvars, nvars, gamma);
    gpu_ns2d_left_eigenvectors(uavg, L, gamma, nvars, dir);
    gpu_ns2d_right_eigenvectors(uavg, R, gamma, nvars, dir);

    gpu_ns2d_matvecmult(nvars, ucL, L, uL + p*nvars);
    gpu_ns2d_matvecmult(nvars, ucR, L, uR + p*nvars);
    gpu_ns2d_matvecmult(nvars, fcL, L, fL + p*nvars);
    gpu_ns2d_matvecmult(nvars, fcR, L, fR + p*nvars);

    double eigL[10], eigC[10], eigR[10];
    gpu_ns2d_eigenvalues(uL + p*nvars, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigL[k] = D[k*nvars+k];
    gpu_ns2d_eigenvalues(uR + p*nvars, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigR[k] = D[k*nvars+k];
    gpu_ns2d_eigenvalues(uavg, D, gamma, nvars, dir);
    for (int k = 0; k < nvars; k++) eigC[k] = D[k*nvars+k];

    for (int k = 0; k < nvars; k++) {
      double alpha = gpu_ns2d_max3(gpu_ns2d_absolute(eigL[k]), gpu_ns2d_absolute(eigC[k]), gpu_ns2d_absolute(eigR[k]));
      fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k] - ucR[k]));
    }

    gpu_ns2d_matvecmult(nvars, fI + p*nvars, R, fc);
  }
}

/* Kernel: Rusanov upwinding for NavierStokes2D */
GPU_KERNEL void gpu_ns2d_upwind_rusanov_kernel(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_interfaces) {
    int indexI[2];
    int temp = idx;
    for (int i = ndims - 1; i >= 0; i--) {
      indexI[i] = temp % bounds_inter[i];
      temp /= bounds_inter[i];
    }

    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }

    int indexL[2], indexR[2];
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

    const int max_nvars = 10;
    double uavg[10], udiff[10];
    if (nvars > max_nvars) return;

    for (int k = 0; k < nvars; k++) {
      udiff[k] = 0.5 * (uR[p*nvars+k] - uL[p*nvars+k]);
    }

    gpu_ns2d_roe_average(uavg, u + pL*nvars, u + pR*nvars, nvars, gamma);

    /* Compute wave speeds */
    double rho, vel[2], e, P, c, vsq;

    /* Left state */
    rho = u[pL*nvars + 0];
    vel[0] = (rho == 0) ? 0 : u[pL*nvars + 1] / rho;
    vel[1] = (rho == 0) ? 0 : u[pL*nvars + 2] / rho;
    e = u[pL*nvars + 3];
    vsq = vel[0]*vel[0] + vel[1]*vel[1];
    P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    c = sqrt(gamma * P / rho);
    double alphaL = c + gpu_ns2d_absolute(vel[dir]);

    /* Right state */
    rho = u[pR*nvars + 0];
    vel[0] = (rho == 0) ? 0 : u[pR*nvars + 1] / rho;
    vel[1] = (rho == 0) ? 0 : u[pR*nvars + 2] / rho;
    e = u[pR*nvars + 3];
    vsq = vel[0]*vel[0] + vel[1]*vel[1];
    P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    c = sqrt(gamma * P / rho);
    double alphaR = c + gpu_ns2d_absolute(vel[dir]);

    /* Average state */
    rho = uavg[0];
    vel[0] = (rho == 0) ? 0 : uavg[1] / rho;
    vel[1] = (rho == 0) ? 0 : uavg[2] / rho;
    e = uavg[3];
    vsq = vel[0]*vel[0] + vel[1]*vel[1];
    P = (e - 0.5*rho*vsq) * (gamma - 1.0);
    c = sqrt(gamma * P / rho);
    double alphaavg = c + gpu_ns2d_absolute(vel[dir]);

    double alpha = gpu_ns2d_max3(alphaL, alphaR, alphaavg);

    for (int k = 0; k < nvars; k++) {
      fI[p*nvars+k] = 0.5 * (fL[p*nvars+k] + fR[p*nvars+k]) - alpha * udiff[k];
    }
  }
}
