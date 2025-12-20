/*! @file gpu_interpolation.cu
    @brief GPU kernels for interpolation functions
*/

#include <gpu.h>
#include <physicalmodels/gpu_euler1d_helpers.h>
#include <physicalmodels/gpu_ns2d_helpers.h>
#include <physicalmodels/gpu_ns3d_helpers.h>
#include <math.h>

/* Helper: Unified Roe average dispatch based on ndims */
static __device__ __forceinline__ void gpu_roe_average(
  double *uavg, const double *uL, const double *uR,
  int nvars, int ndims, double gamma
) {
  if (ndims == 1) {
    gpu_euler1d_roe_average(uavg, uL, uR, nvars, gamma);
  } else if (ndims == 2) {
    gpu_ns2d_roe_average(uavg, uL, uR, nvars, gamma);
  } else {
    gpu_ns3d_roe_average(uavg, uL, uR, nvars, gamma);
  }
}

/* Helper: Unified left eigenvector dispatch based on ndims */
static __device__ __forceinline__ void gpu_left_eigenvectors(
  const double *u, double *L, double gamma, int nvars, int ndims, int dir
) {
  if (ndims == 1) {
    gpu_euler1d_left_eigenvectors(u, L, gamma, nvars);
  } else if (ndims == 2) {
    gpu_ns2d_left_eigenvectors(u, L, gamma, nvars, dir);
  } else {
    gpu_ns3d_left_eigenvectors(u, L, gamma, nvars, dir);
  }
}

/* Helper: Unified right eigenvector dispatch based on ndims */
static __device__ __forceinline__ void gpu_right_eigenvectors(
  const double *u, double *R, double gamma, int nvars, int ndims, int dir
) {
  if (ndims == 1) {
    gpu_euler1d_right_eigenvectors(u, R, gamma, nvars);
  } else if (ndims == 2) {
    gpu_ns2d_right_eigenvectors(u, R, gamma, nvars, dir);
  } else {
    gpu_ns3d_right_eigenvectors(u, R, gamma, nvars, dir);
  }
}

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* ---- MUSCL2 limiter helpers (component-wise) ---- */
static __device__ __forceinline__ double gpu_min(double a, double b) { return (a < b) ? a : b; }
static __device__ __forceinline__ double gpu_max(double a, double b) { return (a > b) ? a : b; }
static __device__ __forceinline__ double gpu_min3(double a, double b, double c) { return gpu_min(a, gpu_min(b,c)); }
static __device__ __forceinline__ double gpu_max3(double a, double b, double c) { return gpu_max(a, gpu_max(b,c)); }

/* limiter_id mapping (host side in gpu_interpolation_function.c):
   0: generalized minmod (gmm)
   1: minmod
   2: vanleer
   3: superbee */
static __device__ __forceinline__ double gpu_muscl2_phi(double r, int limiter_id)
{
  if (limiter_id == 1) {
    /* minmod */
    return gpu_max(0.0, gpu_min(1.0, r));
  } else if (limiter_id == 2) {
    /* vanleer */
    const double ar = fabs(r);
    return (r + ar) / (1.0 + ar);
  } else if (limiter_id == 3) {
    /* superbee */
    return gpu_max3(0.0, gpu_min(2.0*r, 1.0), gpu_min(r, 2.0));
  } else {
    /* generalized minmod with theta=1 (default in CPU code) */
    const double theta = 1.0;
    return gpu_max(0.0, gpu_min3(theta*r, 0.5*(1.0+r), theta));
  }
}

/* Kernel: 5th order WENO interpolation
   Computes interpolated values at interfaces using WENO weights
*/
GPU_KERNEL void gpu_weno5_interpolation_kernel(
  double *fI,           /* output: interpolated values at interfaces */
  const double *fC,     /* input: cell-centered values */
  const double *w1,      /* input: WENO weight 1 */
  const double *w2,      /* input: WENO weight 2 */
  const double *w3,      /* input: WENO weight 3 */
  int nvars,            /* number of variables */
  int ninterfaces,      /* number of interfaces */
  int stride,           /* stride in fC array */
  int upw               /* upwind direction: >0 left, <0 right */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ninterfaces) {
    /* Define stencil points based on upwind direction */
    int qm1, qm2, qm3, qp1, qp2;
    if (upw > 0) {
      /* Left-biased: interface at i-1/2 */
      qm1 = idx - 1 + stride;  /* i-1 */
      qm3 = qm1 - 2*stride;
      qm2 = qm1 - stride;
      qp1 = qm1 + stride;
      qp2 = qm1 + 2*stride;
    } else {
      /* Right-biased: interface at i+1/2 */
      qm1 = idx + stride;  /* i */
      qm3 = qm1 + 2*stride;
      qm2 = qm1 + stride;
      qp1 = qm1 - stride;
      qp2 = qm1 - 2*stride;
    }
    
    /* Candidate stencils */
    static const double one_sixth = 1.0/6.0;
    
    for (int v = 0; v < nvars; v++) {
      /* Stencil 1: i-3, i-2, i-1 */
      double f1 = (2*one_sixth)*fC[qm3*nvars+v] + (-7*one_sixth)*fC[qm2*nvars+v] + (11*one_sixth)*fC[qm1*nvars+v];
      /* Stencil 2: i-2, i-1, i */
      double f2 = (-one_sixth)*fC[qm2*nvars+v] + (5*one_sixth)*fC[qm1*nvars+v] + (2*one_sixth)*fC[qp1*nvars+v];
      /* Stencil 3: i-1, i, i+1 */
      double f3 = (2*one_sixth)*fC[qm1*nvars+v] + (5*one_sixth)*fC[qp1*nvars+v] + (-one_sixth)*fC[qp2*nvars+v];
      
      /* Weighted combination */
      fI[idx*nvars+v] = w1[idx*nvars+v]*f1 + w2[idx*nvars+v]*f2 + w3[idx*nvars+v]*f3;
    }
  }
}

/* Kernel: 2nd order central interpolation */
GPU_KERNEL void gpu_central2_interpolation_kernel(
  double *fI,           /* output: interpolated values at interfaces */
  const double *fC,     /* input: cell-centered values */
  int nvars,            /* number of variables */
  int ninterfaces,      /* number of interfaces */
  int stride            /* stride in fC array */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ninterfaces) {
    int q1 = idx + stride;      /* cell i */
    int q2 = idx + stride + 1;  /* cell i+1 */
    
    for (int v = 0; v < nvars; v++) {
      fI[idx*nvars+v] = 0.5 * (fC[q1*nvars+v] + fC[q2*nvars+v]);
    }
  }
}

/* Kernel: 4th order central interpolation */
GPU_KERNEL void gpu_central4_interpolation_kernel(
  double *fI,           /* output: interpolated values at interfaces */
  const double *fC,     /* input: cell-centered values */
  int nvars,            /* number of variables */
  int ninterfaces,      /* number of interfaces */
  int stride            /* stride in fC array */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ninterfaces) {
    int qm1 = idx - 1 + stride;  /* cell i-1 */
    int q0  = idx + stride;       /* cell i */
    int qp1 = idx + 1 + stride;  /* cell i+1 */
    int qp2 = idx + 2 + stride;  /* cell i+2 */
    
    static const double c0 = -1.0/12.0;
    static const double c1 = 7.0/12.0;
    static const double c2 = 7.0/12.0;
    static const double c3 = -1.0/12.0;
    
    for (int v = 0; v < nvars; v++) {
      fI[idx*nvars+v] = c0*fC[qm1*nvars+v] + c1*fC[q0*nvars+v] + c2*fC[qp1*nvars+v] + c3*fC[qp2*nvars+v];
    }
  }
}

/* Kernel: 3rd order MUSCL interpolation with Koren's limiter */
GPU_KERNEL void gpu_muscl3_interpolation_kernel(
  double *fI,           /* output: interpolated values at interfaces */
  const double *fC,     /* input: cell-centered values */
  int nvars,            /* number of variables */
  int ninterfaces,      /* number of interfaces */
  int stride,           /* stride in fC array */
  int upw,              /* upwind direction */
  double eps            /* epsilon for limiter */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ninterfaces) {
    int qm2, qm1, q0;
    if (upw > 0) {
      /* Left-biased */
      qm2 = idx - 2 + stride;
      qm1 = idx - 1 + stride;
      q0  = idx + stride;
    } else {
      /* Right-biased */
      qm2 = idx + 2 + stride;
      qm1 = idx + 1 + stride;
      q0  = idx + stride;
    }
    
    static const double one_third = 1.0/3.0;
    static const double one_sixth = 1.0/6.0;
    
    for (int v = 0; v < nvars; v++) {
      double df1 = fC[qm1*nvars+v] - fC[qm2*nvars+v];
      double df2 = fC[q0*nvars+v] - fC[qm1*nvars+v];
      
      /* Koren's limiter */
      double num = 3.0 * df1 * df2 + eps;
      double den = 2.0 * (df2 - df1) * (df2 - df1) + 3.0 * df1 * df2 + eps;
      double phi = (den > 1e-14) ? num / den : 1.0;
      
      /* MUSCL interpolation */
      fI[idx*nvars+v] = fC[qm1*nvars+v] + phi * (one_third*df2 + one_sixth*df1);
    }
  }
}

/* Kernel: Multi-dimensional MUSCL2 interpolation (component-wise)
   Mirrors Interp1PrimSecondOrderMUSCL.c indexing & formula. */
GPU_KERNEL void gpu_muscl2_interpolation_nd_kernel(
  double *fI,
  const double *fC,
  int nvars, int ndims,
  const int *dim,
  const int *stride_with_ghosts,
  const int *bounds_inter,
  int ghosts,
  int dir,
  int upw,
  int limiter_id
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];

  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }

  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];

  const int stride_dir = stride_with_ghosts[dir];

  int q_base = 0;
  if (upw > 0) {
    /* qm1 = indexI-1 cell */
    indexC[dir] = indexI[dir] - 1;
  } else {
    /* qp1 = indexI cell */
    indexC[dir] = indexI[dir];
  }

  /* compute base 1D index for that cell center (includes ghosts) */
  q_base = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) q_base = q_base * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);

  int qm1 = 0, qp1 = 0, qm2 = 0, qp2 = 0;
  if (upw > 0) {
    qm1 = q_base;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
  } else {
    qp1 = q_base;
    qp2 = qp1 + stride_dir;
    qm1 = qp1 - stride_dir;
  }

  for (int v = 0; v < nvars; v++) {
    if (upw > 0) {
      const double m2 = fC[qm2*nvars + v];
      const double m1 = fC[qm1*nvars + v];
      const double p1 = fC[qp1*nvars + v];
      const double slope_ratio = (m1 - m2) / ((p1 - m1) + 1e-40);
      const double phi = gpu_muscl2_phi(slope_ratio, limiter_id);
      fI[tid*nvars + v] = m1 + 0.5 * phi * (p1 - m1);
    } else {
      const double m1 = fC[qm1*nvars + v];
      const double p1 = fC[qp1*nvars + v];
      const double p2 = fC[qp2*nvars + v];
      const double slope_ratio = (p1 - m1) / ((p2 - p1) + 1e-40);
      const double phi = gpu_muscl2_phi(slope_ratio, limiter_id);
      fI[tid*nvars + v] = p1 + 0.5 * phi * (p1 - p2);
    }
  }
}

/* Kernel: Multi-dimensional MUSCL3 interpolation (component-wise, Koren limiter form)
   Mirrors Interp1PrimThirdOrderMUSCL.c indexing & formula. */
GPU_KERNEL void gpu_muscl3_interpolation_nd_kernel(
  double *fI,
  const double *fC,
  int nvars, int ndims,
  const int *dim,
  const int *stride_with_ghosts,
  const int *bounds_inter,
  int ghosts,
  int dir,
  int upw,
  double eps
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];

  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }

  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];

  const int stride_dir = stride_with_ghosts[dir];

  static const double one_third = 1.0/3.0;
  static const double one_sixth = 1.0/6.0;

  int qm1 = 0, qm2 = 0, qp1 = 0, qp2 = 0;
  if (upw > 0) {
    /* qm1 cell: indexI-1 */
    indexC[dir] = indexI[dir] - 1;
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
  } else {
    /* qp1 cell: indexI */
    indexC[dir] = indexI[dir];
    qp1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qp1 = qp1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm1 = qp1 - stride_dir;
    qp2 = qp1 + stride_dir;
  }

  for (int v = 0; v < nvars; v++) {
    if (upw > 0) {
      const double m2 = fC[qm2*nvars + v];
      const double m1 = fC[qm1*nvars + v];
      const double p1 = fC[qp1*nvars + v];
      const double fdiff = p1 - m1;
      const double bdiff = m1 - m2;
      const double num = (3.0*fdiff*bdiff + eps);
      const double den = (2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps);
      const double limit = (den != 0.0) ? (num/den) : 1.0;
      fI[tid*nvars + v] = m1 + limit * (one_third*fdiff + one_sixth*bdiff);
    } else {
      const double m1 = fC[qm1*nvars + v];
      const double p1 = fC[qp1*nvars + v];
      const double p2 = fC[qp2*nvars + v];
      const double fdiff = p2 - p1;
      const double bdiff = p1 - m1;
      const double num = (3.0*fdiff*bdiff + eps);
      const double den = (2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps);
      const double limit = (den != 0.0) ? (num/den) : 1.0;
      fI[tid*nvars + v] = p1 - limit * (one_third*fdiff + one_sixth*bdiff);
    }
  }
}

/* Kernel: Multi-dimensional MUSCL2 interpolation (characteristic-based, model-agnostic) */
GPU_KERNEL void gpu_muscl2_interpolation_nd_char_ns3d_kernel(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id,
  double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;

  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  const int stride_dir = stride_with_ghosts[dir];

  int qm1 = 0, qm2 = 0, qp1 = 0, qp2 = 0;
  if (upw > 0) {
    indexC[dir] = indexI[dir] - 1;
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
  } else {
    indexC[dir] = indexI[dir];
    qp1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qp1 = qp1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm1 = qp1 - stride_dir;
    qp2 = qp1 + stride_dir;
  }

  /* Roe-average and eigensystem using unified dispatch */
  double uavg[5];  /* max base_nvars = 5 */
  gpu_roe_average(uavg, u + qm1*nvars, u + qp1*nvars, base_nvars, ndims, gamma);
  double L[25], R[25];  /* max 5x5 */
  gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
  gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);

  double fchar[5];  /* max base_nvars = 5 */
  for (int v = 0; v < base_nvars; v++) {
    double a_m2 = 0.0, a_m1 = 0.0, a_p1 = 0.0, a_p2 = 0.0;
    if (upw > 0) {
      for (int k = 0; k < base_nvars; k++) {
        a_m2 += L[v*base_nvars + k] * fC[qm2*nvars + k];
        a_m1 += L[v*base_nvars + k] * fC[qm1*nvars + k];
        a_p1 += L[v*base_nvars + k] * fC[qp1*nvars + k];
      }
      const double r = (a_m1 - a_m2) / ((a_p1 - a_m1) + 1e-40);
      const double phi = gpu_muscl2_phi(r, limiter_id);
      fchar[v] = a_m1 + 0.5 * phi * (a_p1 - a_m1);
    } else {
      for (int k = 0; k < base_nvars; k++) {
        a_m1 += L[v*base_nvars + k] * fC[qm1*nvars + k];
        a_p1 += L[v*base_nvars + k] * fC[qp1*nvars + k];
        a_p2 += L[v*base_nvars + k] * fC[qp2*nvars + k];
      }
      const double r = (a_p1 - a_m1) / ((a_p2 - a_p1) + 1e-40);
      const double phi = gpu_muscl2_phi(r, limiter_id);
      fchar[v] = a_p1 + 0.5 * phi * (a_p1 - a_p2);
    }
  }
  /* Back-transform base vars */
  for (int k = 0; k < base_nvars; k++) {
    double s = 0.0;
    for (int v = 0; v < base_nvars; v++) s += R[k*base_nvars + v] * fchar[v];
    fI[tid*nvars + k] = s;
  }
  /* Passive scalars: component-wise MUSCL2 */
  for (int k = base_nvars; k < nvars; k++) {
    if (upw > 0) {
      const double m2 = fC[qm2*nvars + k];
      const double m1 = fC[qm1*nvars + k];
      const double p1 = fC[qp1*nvars + k];
      const double r = (m1 - m2) / ((p1 - m1) + 1e-40);
      const double phi = gpu_muscl2_phi(r, limiter_id);
      fI[tid*nvars + k] = m1 + 0.5 * phi * (p1 - m1);
    } else {
      const double m1 = fC[qm1*nvars + k];
      const double p1 = fC[qp1*nvars + k];
      const double p2 = fC[qp2*nvars + k];
      const double r = (p1 - m1) / ((p2 - p1) + 1e-40);
      const double phi = gpu_muscl2_phi(r, limiter_id);
      fI[tid*nvars + k] = p1 + 0.5 * phi * (p1 - p2);
    }
  }
}

/* Kernel: Multi-dimensional MUSCL3 interpolation (characteristic-based, model-agnostic) */
GPU_KERNEL void gpu_muscl3_interpolation_nd_char_ns3d_kernel(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps,
  double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;

  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  const int stride_dir = stride_with_ghosts[dir];

  static const double one_third = 1.0/3.0;
  static const double one_sixth = 1.0/6.0;

  int qm1 = 0, qm2 = 0, qp1 = 0, qp2 = 0;
  if (upw > 0) {
    indexC[dir] = indexI[dir] - 1;
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
  } else {
    indexC[dir] = indexI[dir];
    qp1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qp1 = qp1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm1 = qp1 - stride_dir;
    qp2 = qp1 + stride_dir;
  }

  /* Roe-average and eigensystem using unified dispatch */
  double uavg[5];  /* max base_nvars = 5 */
  gpu_roe_average(uavg, u + qm1*nvars, u + qp1*nvars, base_nvars, ndims, gamma);
  double L[25], R[25];  /* max 5x5 */
  gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
  gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);

  double fchar[5];  /* max base_nvars = 5 */
  for (int v = 0; v < base_nvars; v++) {
    double a_m2 = 0.0, a_m1 = 0.0, a_p1 = 0.0, a_p2 = 0.0;
    if (upw > 0) {
      for (int k = 0; k < base_nvars; k++) {
        a_m2 += L[v*base_nvars + k] * fC[qm2*nvars + k];
        a_m1 += L[v*base_nvars + k] * fC[qm1*nvars + k];
        a_p1 += L[v*base_nvars + k] * fC[qp1*nvars + k];
      }
      const double fdiff = a_p1 - a_m1;
      const double bdiff = a_m1 - a_m2;
      const double num = (3.0*fdiff*bdiff + eps);
      const double den = (2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps);
      const double limit = (den != 0.0) ? (num/den) : 1.0;
      fchar[v] = a_m1 + limit * (one_third*fdiff + one_sixth*bdiff);
    } else {
      for (int k = 0; k < base_nvars; k++) {
        a_m1 += L[v*base_nvars + k] * fC[qm1*nvars + k];
        a_p1 += L[v*base_nvars + k] * fC[qp1*nvars + k];
        a_p2 += L[v*base_nvars + k] * fC[qp2*nvars + k];
      }
      const double fdiff = a_p2 - a_p1;
      const double bdiff = a_p1 - a_m1;
      const double num = (3.0*fdiff*bdiff + eps);
      const double den = (2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps);
      const double limit = (den != 0.0) ? (num/den) : 1.0;
      fchar[v] = a_p1 - limit * (one_third*fdiff + one_sixth*bdiff);
    }
  }
  /* Back-transform base vars */
  for (int k = 0; k < base_nvars; k++) {
    double s = 0.0;
    for (int v = 0; v < base_nvars; v++) s += R[k*base_nvars + v] * fchar[v];
    fI[tid*nvars + k] = s;
  }
  /* Passive scalars: component-wise MUSCL3 */
  for (int k = base_nvars; k < nvars; k++) {
    if (upw > 0) {
      const double m2 = fC[qm2*nvars + k];
      const double m1 = fC[qm1*nvars + k];
      const double p1 = fC[qp1*nvars + k];
      const double fdiff = p1 - m1;
      const double bdiff = m1 - m2;
      const double num = (3.0*fdiff*bdiff + eps);
      const double den = (2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps);
      const double limit = (den != 0.0) ? (num/den) : 1.0;
      fI[tid*nvars + k] = m1 + limit * (one_third*fdiff + one_sixth*bdiff);
    } else {
      const double m1 = fC[qm1*nvars + k];
      const double p1 = fC[qp1*nvars + k];
      const double p2 = fC[qp2*nvars + k];
      const double fdiff = p2 - p1;
      const double bdiff = p1 - m1;
      const double num = (3.0*fdiff*bdiff + eps);
      const double den = (2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps);
      const double limit = (den != 0.0) ? (num/den) : 1.0;
      fI[tid*nvars + k] = p1 - limit * (one_third*fdiff + one_sixth*bdiff);
    }
  }
}

/* Kernel: Multi-dimensional 5th order WENO interpolation
   Handles full multi-dimensional array layout using stride_with_ghosts
*/
GPU_KERNEL void gpu_weno5_interpolation_nd_kernel(
  double *fI,                    /* output: interpolated values at interfaces (no ghosts) */
  const double *fC,              /* input: cell-centered values (with ghosts) */
  const double *w1,              /* input: WENO weight 1 */
  const double *w2,              /* input: WENO weight 2 */
  const double *w3,              /* input: WENO weight 3 */
  int nvars,                     /* number of variables */
  int ndims,                     /* number of dimensions */
  const int *dim,                /* dimension sizes (without ghosts) */
  const int *stride_with_ghosts, /* stride array for cell-centered arrays */
  const int *bounds_inter,       /* bounds for interface array */
  int ghosts,                    /* number of ghost points */
  int dir,                       /* direction along which to interpolate */
  int upw                        /* upwind direction: >0 left, <0 right */
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
    
    /* Compute 1D index for interface point (no ghosts) - matches _ArrayIndex1D_ with ghost=0 */
    int p = indexI[ndims-1];
    for (int i = ndims-2; i >= 0; i--) {
      p = p * bounds_inter[i] + indexI[i];
    }
    
    /* Compute cell-centered index for this interface */
    int indexC[3];
    for (int i = 0; i < ndims; i++) {
      indexC[i] = indexI[i];
    }
    
    /* Compute stencil point indices - match CPU code exactly */
    /* CPU code computes qm1 first, then uses relative offsets with stride[dir] */
    int qm1, qm2, qm3, qp1, qp2;
    if (upw > 0) {
      /* Left-biased: interface at i-1/2 */
      /* Compute qm1 first (matching _ArrayIndex1D_ pattern) */
      indexC[dir] = indexI[dir] - 1;
      qm1 = indexC[ndims-1] + ghosts;
      for (int i = ndims-2; i >= 0; i--) {
        qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
      }
      /* Use relative offsets with stride_with_ghosts[dir] (matching CPU code) */
      qm3 = qm1 - 2*stride_with_ghosts[dir];
      qm2 = qm1 -   stride_with_ghosts[dir];
      qp1 = qm1 +   stride_with_ghosts[dir];
      qp2 = qm1 + 2*stride_with_ghosts[dir];
    } else {
      /* Right-biased: interface at i+1/2 */
      /* Compute qm1 first (matching _ArrayIndex1D_ pattern) */
      indexC[dir] = indexI[dir];
      qm1 = indexC[ndims-1] + ghosts;
      for (int i = ndims-2; i >= 0; i--) {
        qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
      }
      /* Use relative offsets with stride_with_ghosts[dir] (matching CPU code) */
      qm3 = qm1 + 2*stride_with_ghosts[dir];
      qm2 = qm1 +   stride_with_ghosts[dir];
      qp1 = qm1 -   stride_with_ghosts[dir];
      qp2 = qm1 - 2*stride_with_ghosts[dir];
    }
    
    /* Candidate stencils */
    static const double one_sixth = 1.0/6.0;
    
    for (int v = 0; v < nvars; v++) {
      /* Stencil 1: i-3, i-2, i-1 */
      double f1 = (2*one_sixth)*fC[qm3*nvars+v] + (-7*one_sixth)*fC[qm2*nvars+v] + (11*one_sixth)*fC[qm1*nvars+v];
      /* Stencil 2: i-2, i-1, i */
      double f2 = (-one_sixth)*fC[qm2*nvars+v] + (5*one_sixth)*fC[qm1*nvars+v] + (2*one_sixth)*fC[qp1*nvars+v];
      /* Stencil 3: i-1, i, i+1 */
      double f3 = (2*one_sixth)*fC[qm1*nvars+v] + (5*one_sixth)*fC[qp1*nvars+v] + (-one_sixth)*fC[qp2*nvars+v];
      
      /* Weighted combination */
      fI[p*nvars+v] = w1[p*nvars+v]*f1 + w2[p*nvars+v]*f2 + w3[p*nvars+v]*f3;
    }
  }
}

/* Launch wrapper for multi-dimensional WENO5 interpolation */
#define DEFAULT_BLOCK_SIZE 256

extern "C" {
void gpu_launch_weno5_interpolation_nd(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - simplified version */
  static const double one_sixth = 1.0/6.0;
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  for (int idx = 0; idx < total_interfaces; idx++) {
    /* Simplified CPU version - would need full index computation */
  }
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  
  /* Copy dim, stride_with_ghosts, and bounds_inter to GPU if needed */
  int *dim_gpu = NULL;
  int *stride_gpu = NULL;
  int *bounds_inter_gpu = NULL;
  
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate dim_gpu for interpolation\n");
    return;
  }
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate stride_gpu for interpolation\n");
    GPUFree(dim_gpu);
    return;
  }
  if (GPUAllocate((void**)&bounds_inter_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate bounds_inter_gpu for interpolation\n");
    GPUFree(dim_gpu);
    GPUFree(stride_gpu);
    return;
  }
  
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(bounds_inter_gpu, bounds_inter, ndims * sizeof(int));
  
  /* Compute total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  
  GPU_KERNEL_LAUNCH(gpu_weno5_interpolation_nd_kernel, gridSize, blockSize)(
    fI, fC, w1, w2, w3, nvars, ndims, dim_gpu, stride_gpu, bounds_inter_gpu, ghosts, dir, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  
  GPUFree(dim_gpu);
  GPUFree(stride_gpu);
  GPUFree(bounds_inter_gpu);
#endif
}

/* Kernel: Multi-dimensional 5th order WENO interpolation (characteristic-based)
   Handles full multi-dimensional array layout using stride_with_ghosts
   Requires solution array u for computing averaged state and eigenvectors
*/
GPU_KERNEL void gpu_weno5_interpolation_nd_char_kernel(
  double *fI,                    /* output: interpolated values at interfaces (no ghosts) */
  const double *fC,              /* input: cell-centered flux values (with ghosts) */
  const double *u,               /* input: cell-centered solution values (with ghosts) */
  const double *w1,              /* input: WENO weight 1 */
  const double *w2,              /* input: WENO weight 2 */
  const double *w3,              /* input: WENO weight 3 */
  int nvars,                     /* number of variables */
  int ndims,                     /* number of dimensions */
  const int *dim,                /* dimension sizes (without ghosts) */
  const int *stride_with_ghosts, /* stride array for cell-centered arrays */
  const int *bounds_inter,       /* bounds for interface array */
  int ghosts,                    /* number of ghost points */
  int dir,                       /* direction along which to interpolate */
  int upw,                       /* upwind direction: >0 left, <0 right */
  double gamma                   /* ratio of heat capacities (for NavierStokes3D) */
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
    
    /* Compute cell-centered indices for this interface */
    int indexC[3];
    for (int i = 0; i < ndims; i++) {
      indexC[i] = indexI[i];
    }
    
    /* Compute stencil point indices - match CPU code exactly */
    /* Compute stencil point indices - match CPU code exactly */
    /* CPU code computes qm1 first, then uses relative offsets with stride[dir] */
    int qm1, qm2, qm3, qp1, qp2;
    int qL, qR; /* Left and right cell indices for averaging */
    if (upw > 0) {
      /* Left-biased: interface at i-1/2 */
      /* Compute qm1 first (matching _ArrayIndex1D_ pattern) */
      indexC[dir] = indexI[dir] - 1;
      qm1 = indexC[ndims-1] + ghosts;
      for (int i = ndims-2; i >= 0; i--) {
        qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
      }
      qL = qm1; /* Left cell for averaging */
      /* Use relative offsets with stride_with_ghosts[dir] (matching CPU code) */
      qm3 = qm1 - 2*stride_with_ghosts[dir];
      qm2 = qm1 -   stride_with_ghosts[dir];
      qp1 = qm1 +   stride_with_ghosts[dir];
      qp2 = qm1 + 2*stride_with_ghosts[dir];
      /* Compute qR for averaging */
      indexC[dir] = indexI[dir];
      qR = indexC[ndims-1] + ghosts;
      for (int i = ndims-2; i >= 0; i--) {
        qR = qR * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
      }
    } else {
      /* Right-biased: interface at i+1/2 */
      /* Compute qm1 first (matching _ArrayIndex1D_ pattern) */
      indexC[dir] = indexI[dir];
      qm1 = indexC[ndims-1] + ghosts;
      for (int i = ndims-2; i >= 0; i--) {
        qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
      }
      qL = qm1; /* Left cell for averaging */
      /* Use relative offsets with stride_with_ghosts[dir] (matching CPU code) */
      qm3 = qm1 + 2*stride_with_ghosts[dir];
      qm2 = qm1 +   stride_with_ghosts[dir];
      qp1 = qm1 -   stride_with_ghosts[dir];
      qp2 = qm1 - 2*stride_with_ghosts[dir];
      /* Compute qR for averaging */
      indexC[dir] = indexI[dir] - 1;
      qR = indexC[ndims-1] + ghosts;
      for (int i = ndims-2; i >= 0; i--) {
        qR = qR * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
      }
    }
    
    /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
    const int base_nvars = ndims + 2;
    
    /* Compute Roe-averaged state at interface for base variables only */
    double uavg[5];  /* max base_nvars = 5 */
    gpu_roe_average(uavg, u + qL*nvars, u + qR*nvars, base_nvars, ndims, gamma);
    
    /* Get left and right eigenvectors at averaged state for base variables only */
    double L[25], R[25];  /* max 5x5 */
    gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
    gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);
    
    /* Candidate stencils */
    static const double one_sixth = 1.0/6.0;
    
    /* Characteristic decomposition for base flow variables */
    double fchar[5];  /* max base_nvars = 5 */
    for (int v = 0; v < base_nvars; v++) {
      /* Transform flux to characteristic space */
      double fm3 = 0.0, fm2 = 0.0, fm1 = 0.0, fp1 = 0.0, fp2 = 0.0;
      for (int k = 0; k < base_nvars; k++) {
        fm3 += L[v*base_nvars+k] * fC[qm3*nvars+k];
        fm2 += L[v*base_nvars+k] * fC[qm2*nvars+k];
        fm1 += L[v*base_nvars+k] * fC[qm1*nvars+k];
        fp1 += L[v*base_nvars+k] * fC[qp1*nvars+k];
        fp2 += L[v*base_nvars+k] * fC[qp2*nvars+k];
      }
      
      /* Candidate stencils */
      double f1 = (2*one_sixth)*fm3 + (-7*one_sixth)*fm2 + (11*one_sixth)*fm1;
      double f2 = (-one_sixth)*fm2 + (5*one_sixth)*fm1 + (2*one_sixth)*fp1;
      double f3 = (2*one_sixth)*fm1 + (5*one_sixth)*fp1 + (-one_sixth)*fp2;
      
      /* WENO weights */
      double w1_val = w1[p*nvars+v];
      double w2_val = w2[p*nvars+v];
      double w3_val = w3[p*nvars+v];
      
      /* Fifth order WENO approximation in characteristic space */
      fchar[v] = w1_val*f1 + w2_val*f2 + w3_val*f3;
    }
    
    /* Transform back to physical space for base variables */
    for (int k = 0; k < base_nvars; k++) {
      double s = 0.0;
      for (int v = 0; v < base_nvars; v++) {
        s += R[k*base_nvars + v] * fchar[v];
      }
      fI[p*nvars + k] = s;
    }
    
    /* Component-wise WENO5 for passive scalars (chemistry variables) */
    for (int k = base_nvars; k < nvars; k++) {
      /* Get stencil values */
      double fm3 = fC[qm3*nvars + k];
      double fm2 = fC[qm2*nvars + k];
      double fm1 = fC[qm1*nvars + k];
      double fp1 = fC[qp1*nvars + k];
      double fp2 = fC[qp2*nvars + k];
      
      /* Candidate stencils */
      double f1 = (2*one_sixth)*fm3 + (-7*one_sixth)*fm2 + (11*one_sixth)*fm1;
      double f2 = (-one_sixth)*fm2 + (5*one_sixth)*fm1 + (2*one_sixth)*fp1;
      double f3 = (2*one_sixth)*fm1 + (5*one_sixth)*fp1 + (-one_sixth)*fp2;
      
      /* WENO weights */
      double w1_val = w1[p*nvars+k];
      double w2_val = w2[p*nvars+k];
      double w3_val = w3[p*nvars+k];
      
      /* Fifth order WENO approximation */
      fI[p*nvars + k] = w1_val*f1 + w2_val*f2 + w3_val*f3;
    }
  }
}

/* Launch wrapper for multi-dimensional WENO5 characteristic-based interpolation */
void gpu_launch_weno5_interpolation_nd_char(
  double *fI, const double *fC, const double *u, const double *w1, const double *w2, const double *w3,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback */
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  
  /* Copy dim, stride_with_ghosts, and bounds_inter to GPU if needed */
  int *dim_gpu = NULL;
  int *stride_gpu = NULL;
  int *bounds_inter_gpu = NULL;
  
  if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate dim_gpu for characteristic interpolation\n");
    return;
  }
  if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate stride_gpu for characteristic interpolation\n");
    GPUFree(dim_gpu);
    return;
  }
  if (GPUAllocate((void**)&bounds_inter_gpu, ndims * sizeof(int))) {
    fprintf(stderr, "Error: Failed to allocate bounds_inter_gpu for characteristic interpolation\n");
    GPUFree(dim_gpu);
    GPUFree(stride_gpu);
    return;
  }
  
  GPUCopyToDevice(dim_gpu, dim, ndims * sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims * sizeof(int));
  GPUCopyToDevice(bounds_inter_gpu, bounds_inter, ndims * sizeof(int));
  
  /* Compute total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }
  
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  
  GPU_KERNEL_LAUNCH(gpu_weno5_interpolation_nd_char_kernel, gridSize, blockSize)(
    fI, fC, u, w1, w2, w3, nvars, ndims, dim_gpu, stride_gpu, bounds_inter_gpu, ghosts, dir, upw, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  
  GPUFree(dim_gpu);
  GPUFree(stride_gpu);
  GPUFree(bounds_inter_gpu);
#endif
}
} /* extern "C" */

/* Launch wrapper for multi-dimensional MUSCL2 interpolation */
extern "C" void gpu_launch_muscl2_interpolation_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id,
  int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)limiter_id; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl2_interpolation_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, limiter_id
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

/* Launch wrapper for multi-dimensional MUSCL3 interpolation */
extern "C" void gpu_launch_muscl3_interpolation_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps,
  int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)eps; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, eps
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_muscl2_interpolation_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id,
  double gamma,
  int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)limiter_id; (void)gamma; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl2_interpolation_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, limiter_id, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_muscl3_interpolation_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps,
  double gamma,
  int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)eps; (void)gamma; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, eps, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

/* ==========================================================================
   New interpolation schemes: first_order_upwind, second_order_central,
   fourth_order_central, fifth_order_upwind (multi-dimensional versions)
   ========================================================================== */

/* Kernel: Multi-dimensional first order upwind interpolation (component-wise) */
GPU_KERNEL void gpu_first_order_upwind_nd_kernel(
  double *fI,
  const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  /* Select cell based on upwind direction */
  if (upw > 0) {
    indexC[dir] = indexI[dir] - 1; /* left cell */
  } else {
    indexC[dir] = indexI[dir];     /* right cell */
  }
  
  /* Compute 1D index for cell center */
  int q = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) q = q * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
  
  for (int v = 0; v < nvars; v++) {
    fI[tid*nvars + v] = fC[q*nvars + v];
  }
}

/* Kernel: Multi-dimensional second order central interpolation (component-wise) */
GPU_KERNEL void gpu_second_order_central_nd_kernel(
  double *fI,
  const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  /* Compute qL (cell at indexI-1) */
  indexC[dir] = indexI[dir] - 1;
  int qL = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) qL = qL * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
  
  /* qR = qL + stride_dir */
  int qR = qL + stride_dir;
  
  for (int v = 0; v < nvars; v++) {
    fI[tid*nvars + v] = 0.5 * (fC[qL*nvars + v] + fC[qR*nvars + v]);
  }
}

/* Kernel: Multi-dimensional fourth order central interpolation (component-wise) */
GPU_KERNEL void gpu_fourth_order_central_nd_kernel(
  double *fI,
  const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  /* Compute qL (cell at indexI-1) */
  indexC[dir] = indexI[dir] - 1;
  int qL = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) qL = qL * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
  
  int qLL = qL - stride_dir;
  int qR = qL + stride_dir;
  int qRR = qL + 2*stride_dir;
  
  static const double c1 = 7.0 / 12.0;
  static const double c2 = -1.0 / 12.0;
  
  for (int v = 0; v < nvars; v++) {
    fI[tid*nvars + v] = c2*fC[qLL*nvars + v] + c1*fC[qL*nvars + v] 
                      + c1*fC[qR*nvars + v] + c2*fC[qRR*nvars + v];
  }
}

/* Kernel: Multi-dimensional fifth order upwind interpolation (component-wise) */
GPU_KERNEL void gpu_fifth_order_upwind_nd_kernel(
  double *fI,
  const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  /* Decode tid to indexI using CPU-compatible ordering (first index fastest) */
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  static const double c1 = 1.0/30.0;
  static const double c2 = -13.0/60.0;
  static const double c3 = 47.0/60.0;
  static const double c4 = 27.0/60.0;
  static const double c5 = -1.0/20.0;
  
  int qm1, qm2, qm3, qp1, qp2;
  if (upw > 0) {
    /* Left-biased */
    indexC[dir] = indexI[dir] - 1;
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm3 = qm1 - 2*stride_dir;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
    qp2 = qm1 + 2*stride_dir;
  } else {
    /* Right-biased (reflected) */
    indexC[dir] = indexI[dir];
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm3 = qm1 + 2*stride_dir;
    qm2 = qm1 + stride_dir;
    qp1 = qm1 - stride_dir;
    qp2 = qm1 - 2*stride_dir;
  }
  
  for (int v = 0; v < nvars; v++) {
    fI[tid*nvars + v] = c1*fC[qm3*nvars + v] + c2*fC[qm2*nvars + v] + c3*fC[qm1*nvars + v]
                      + c4*fC[qp1*nvars + v] + c5*fC[qp2*nvars + v];
  }
}

/* Kernel: Multi-dimensional first order upwind characteristic interpolation (NS3D) */
GPU_KERNEL void gpu_first_order_upwind_nd_char_ns3d_kernel(
  double *fI,
  const double *fC,
  const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  /* Compute qL and qR for averaging */
  indexC[dir] = indexI[dir] - 1;
  int qL = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) qL = qL * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
  int qR = qL + stride_dir;
  
  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;
  
  /* Compute Roe average and eigenvectors using unified dispatch */
  double uavg[5], L[25], R[25];  /* max base_nvars = 5 */
  gpu_roe_average(uavg, &u[qL*nvars], &u[qR*nvars], base_nvars, ndims, gamma);
  gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
  gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);
  
  /* Select upwind cell */
  int q = (upw > 0) ? qL : qR;
  
  /* Transform to characteristic, copy, transform back */
  double fchar[5];
  for (int v = 0; v < base_nvars; v++) {
    fchar[v] = 0.0;
    for (int k = 0; k < base_nvars; k++) {
      fchar[v] += L[v*base_nvars+k] * fC[q*nvars+k];
    }
  }
  
  /* Transform back */
  for (int v = 0; v < base_nvars; v++) {
    double sum = 0.0;
    for (int k = 0; k < base_nvars; k++) sum += R[v*base_nvars+k] * fchar[k];
    fI[tid*nvars + v] = sum;
  }
  
  /* Copy passive scalars directly */
  for (int v = base_nvars; v < nvars; v++) {
    fI[tid*nvars + v] = fC[q*nvars + v];
  }
}

/* Kernel: Multi-dimensional second order central characteristic interpolation (NS3D) */
GPU_KERNEL void gpu_second_order_central_nd_char_ns3d_kernel(
  double *fI,
  const double *fC,
  const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  /* Compute qL and qR */
  indexC[dir] = indexI[dir] - 1;
  int qL = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) qL = qL * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
  int qR = qL + stride_dir;
  
  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;
  
  /* Compute Roe average and eigenvectors using unified dispatch */
  double uavg[5], L[25], R[25];  /* max base_nvars = 5 */
  gpu_roe_average(uavg, &u[qL*nvars], &u[qR*nvars], base_nvars, ndims, gamma);
  gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
  gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);
  
  /* Transform to characteristic space and interpolate */
  double fcharL[5], fcharR[5];
  for (int v = 0; v < base_nvars; v++) {
    fcharL[v] = 0.0; fcharR[v] = 0.0;
    for (int k = 0; k < base_nvars; k++) {
      fcharL[v] += L[v*base_nvars+k] * fC[qL*nvars+k];
      fcharR[v] += L[v*base_nvars+k] * fC[qR*nvars+k];
    }
  }
  
  /* Second order central: average */
  double fchar[5];
  for (int v = 0; v < base_nvars; v++) fchar[v] = 0.5*(fcharL[v] + fcharR[v]);
  
  /* Transform back */
  for (int v = 0; v < base_nvars; v++) {
    double sum = 0.0;
    for (int k = 0; k < base_nvars; k++) sum += R[v*base_nvars+k] * fchar[k];
    fI[tid*nvars + v] = sum;
  }
  
  /* Passive scalars */
  for (int v = base_nvars; v < nvars; v++) {
    fI[tid*nvars + v] = 0.5*(fC[qL*nvars + v] + fC[qR*nvars + v]);
  }
}

/* Kernel: Multi-dimensional fourth order central characteristic interpolation (NS3D) */
GPU_KERNEL void gpu_fourth_order_central_nd_char_ns3d_kernel(
  double *fI,
  const double *fC,
  const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  /* Compute qL (cell at indexI-1) */
  indexC[dir] = indexI[dir] - 1;
  int qL = indexC[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) qL = qL * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
  int qLL = qL - stride_dir;
  int qR = qL + stride_dir;
  int qRR = qL + 2*stride_dir;
  
  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;
  
  /* Compute Roe average and eigenvectors using unified dispatch */
  double uavg[5], L[25], R[25];  /* max base_nvars = 5 */
  gpu_roe_average(uavg, &u[qL*nvars], &u[qR*nvars], base_nvars, ndims, gamma);
  gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
  gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);
  
  static const double c1 = 7.0 / 12.0;
  static const double c2 = -1.0 / 12.0;
  
  /* Transform to characteristic space and interpolate */
  double fcharLL[5], fcharL[5], fcharR[5], fcharRR[5];
  for (int v = 0; v < base_nvars; v++) {
    fcharLL[v] = fcharL[v] = fcharR[v] = fcharRR[v] = 0.0;
    for (int k = 0; k < base_nvars; k++) {
      fcharLL[v] += L[v*base_nvars+k] * fC[qLL*nvars+k];
      fcharL[v]  += L[v*base_nvars+k] * fC[qL*nvars+k];
      fcharR[v]  += L[v*base_nvars+k] * fC[qR*nvars+k];
      fcharRR[v] += L[v*base_nvars+k] * fC[qRR*nvars+k];
    }
  }
  
  /* Fourth order central interpolation in characteristic space */
  double fchar[5];
  for (int v = 0; v < base_nvars; v++) {
    fchar[v] = c2*fcharLL[v] + c1*fcharL[v] + c1*fcharR[v] + c2*fcharRR[v];
  }
  
  /* Transform back */
  for (int v = 0; v < base_nvars; v++) {
    double sum = 0.0;
    for (int k = 0; k < base_nvars; k++) sum += R[v*base_nvars+k] * fchar[k];
    fI[tid*nvars + v] = sum;
  }
  
  /* Passive scalars */
  for (int v = base_nvars; v < nvars; v++) {
    fI[tid*nvars + v] = c2*fC[qLL*nvars + v] + c1*fC[qL*nvars + v] 
                      + c1*fC[qR*nvars + v] + c2*fC[qRR*nvars + v];
  }
}

/* Kernel: Multi-dimensional fifth order upwind characteristic interpolation (NS3D) */
GPU_KERNEL void gpu_fifth_order_upwind_nd_char_ns3d_kernel(
  double *fI,
  const double *fC,
  const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma
)
{
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;
  
  int indexI[3] = {0,0,0};
  int tmp = tid;
  for (int i = 0; i < ndims; i++) { indexI[i] = tmp % bounds_inter[i]; tmp /= bounds_inter[i]; }
  
  int indexC[3];
  for (int i = 0; i < ndims; i++) indexC[i] = indexI[i];
  
  const int stride_dir = stride_with_ghosts[dir];
  
  static const double c1 = 1.0/30.0;
  static const double c2 = -13.0/60.0;
  static const double c3 = 47.0/60.0;
  static const double c4 = 27.0/60.0;
  static const double c5 = -1.0/20.0;
  
  int qL, qR, qm1, qm2, qm3, qp1, qp2;
  if (upw > 0) {
    indexC[dir] = indexI[dir] - 1;
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm3 = qm1 - 2*stride_dir;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
    qp2 = qm1 + 2*stride_dir;
    qL = qm1;
    qR = qp1;
  } else {
    indexC[dir] = indexI[dir];
    qm1 = indexC[ndims-1] + ghosts;
    for (int i = ndims - 2; i >= 0; i--) qm1 = qm1 * (dim[i] + 2*ghosts) + (indexC[i] + ghosts);
    qm3 = qm1 + 2*stride_dir;
    qm2 = qm1 + stride_dir;
    qp1 = qm1 - stride_dir;
    qp2 = qm1 - 2*stride_dir;
    qL = qp1;
    qR = qm1;
  }
  
  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;
  
  /* Compute Roe average and eigenvectors using unified dispatch */
  double uavg[5], L[25], R[25];  /* max base_nvars = 5 */
  gpu_roe_average(uavg, &u[qL*nvars], &u[qR*nvars], base_nvars, ndims, gamma);
  gpu_left_eigenvectors(uavg, L, gamma, base_nvars, ndims, dir);
  gpu_right_eigenvectors(uavg, R, gamma, base_nvars, ndims, dir);
  
  /* Transform to characteristic space */
  double fm3[5], fm2[5], fm1[5], fp1[5], fp2[5];
  for (int v = 0; v < base_nvars; v++) {
    fm3[v] = fm2[v] = fm1[v] = fp1[v] = fp2[v] = 0.0;
    for (int k = 0; k < base_nvars; k++) {
      fm3[v] += L[v*base_nvars+k] * fC[qm3*nvars+k];
      fm2[v] += L[v*base_nvars+k] * fC[qm2*nvars+k];
      fm1[v] += L[v*base_nvars+k] * fC[qm1*nvars+k];
      fp1[v] += L[v*base_nvars+k] * fC[qp1*nvars+k];
      fp2[v] += L[v*base_nvars+k] * fC[qp2*nvars+k];
    }
  }
  
  /* Fifth order upwind interpolation in characteristic space */
  double fchar[5];
  for (int v = 0; v < base_nvars; v++) {
    fchar[v] = c1*fm3[v] + c2*fm2[v] + c3*fm1[v] + c4*fp1[v] + c5*fp2[v];
  }
  
  /* Transform back */
  for (int v = 0; v < base_nvars; v++) {
    double sum = 0.0;
    for (int k = 0; k < base_nvars; k++) sum += R[v*base_nvars+k] * fchar[k];
    fI[tid*nvars + v] = sum;
  }
  
  /* Passive scalars */
  for (int v = base_nvars; v < nvars; v++) {
    fI[tid*nvars + v] = c1*fC[qm3*nvars + v] + c2*fC[qm2*nvars + v] + c3*fC[qm1*nvars + v]
                      + c4*fC[qp1*nvars + v] + c5*fC[qp2*nvars + v];
  }
}

/* ========== Launch wrappers for new interpolation schemes ========== */

extern "C" void gpu_launch_first_order_upwind_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_order_upwind_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_second_order_central_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_second_order_central_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_fourth_order_central_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fourth_order_central_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_fifth_order_upwind_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fifth_order_upwind_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

/* Characteristic launch wrappers */
extern "C" void gpu_launch_first_order_upwind_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)gamma; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_order_upwind_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_second_order_central_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_second_order_central_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_fourth_order_central_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fourth_order_central_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

extern "C" void gpu_launch_fifth_order_upwind_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)gamma; (void)blockSize;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int *dim_gpu = NULL, *stride_gpu = NULL, *bounds_gpu = NULL;
  if (GPUAllocate((void**)&dim_gpu, ndims*sizeof(int))) return;
  if (GPUAllocate((void**)&stride_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); return; }
  if (GPUAllocate((void**)&bounds_gpu, ndims*sizeof(int))) { GPUFree(dim_gpu); GPUFree(stride_gpu); return; }
  GPUCopyToDevice(dim_gpu, dim, ndims*sizeof(int));
  GPUCopyToDevice(stride_gpu, stride_with_ghosts, ndims*sizeof(int));
  GPUCopyToDevice(bounds_gpu, bounds_inter, ndims*sizeof(int));
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fifth_order_upwind_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  GPUFree(dim_gpu); GPUFree(stride_gpu); GPUFree(bounds_gpu);
#endif
}

