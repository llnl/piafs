/*! @file gpu_interpolation.cu
    @brief GPU kernels for interpolation functions
*/

#include <gpu.h>
#include <gpu_config.h>
#include <physicalmodels/gpu_euler1d_helpers.h>
#include <physicalmodels/gpu_ns2d_helpers.h>
#include <physicalmodels/gpu_ns3d_helpers.h>
#include <math.h>

/* Helper struct to manage device arrays - avoids repeated alloc/free */
struct GPUInterpMetadata {
  int *dim_gpu;
  int *stride_gpu;
  int *bounds_gpu;
  int ndims;
  bool allocated;
  bool dim_stride_cached;  /* Track if dim/stride have been copied */
  int cached_dim[3];       /* Cached host-side dim values */
  int cached_stride[3];    /* Cached host-side stride values */

  GPUInterpMetadata() : dim_gpu(NULL), stride_gpu(NULL), bounds_gpu(NULL), ndims(0), allocated(false), dim_stride_cached(false) {
    cached_dim[0] = cached_dim[1] = cached_dim[2] = 0;
    cached_stride[0] = cached_stride[1] = cached_stride[2] = 0;
  }

  int setup(const int *dim_host, const int *stride_host, const int *bounds_host, int nd) {
    /* Only reallocate if ndims changed; otherwise reuse existing device buffers */
    if (!allocated || ndims != nd) {
      cleanup();
      ndims = nd;

      if (GPUAllocate((void**)&dim_gpu, ndims * sizeof(int))) return 1;
      if (GPUAllocate((void**)&stride_gpu, ndims * sizeof(int))) {
        GPUFree(dim_gpu); dim_gpu = NULL;
        return 1;
      }
      if (GPUAllocate((void**)&bounds_gpu, ndims * sizeof(int))) {
        GPUFree(dim_gpu); GPUFree(stride_gpu);
        dim_gpu = NULL; stride_gpu = NULL;
        return 1;
      }
      allocated = true;
      dim_stride_cached = false;
    }

    /* Check if dim or stride have changed - important for multi-simulation runs */
    bool dims_changed = false;
    if (dim_stride_cached) {
      /* Compare with cached host-side values (no device-to-host transfer needed) */
      for (int i = 0; i < ndims; i++) {
        if (cached_dim[i] != dim_host[i] || cached_stride[i] != stride_host[i]) {
          dims_changed = true;
          break;
        }
      }
    }

    /* Copy dim and stride if not cached or if dimensions have changed */
    if (!dim_stride_cached || dims_changed) {
      GPUCopyToDevice(dim_gpu, dim_host, ndims * sizeof(int));
      GPUCopyToDevice(stride_gpu, stride_host, ndims * sizeof(int));
      /* Cache the host-side values */
      for (int i = 0; i < ndims; i++) {
        cached_dim[i] = dim_host[i];
        cached_stride[i] = stride_host[i];
      }
      dim_stride_cached = true;
      #ifdef GPU_INTERP_DEBUG
      fprintf(stderr, "[GPU_INTERP] %s: copied dim and stride (ndims=%d)\n",
              dims_changed ? "Dimensions changed" : "First call", ndims);
      #endif
    }

    /* Always copy bounds_inter - it changes per direction */
    GPUCopyToDevice(bounds_gpu, bounds_host, ndims * sizeof(int));
    #ifdef GPU_INTERP_DEBUG
    fprintf(stderr, "[GPU_INTERP] Copied bounds_inter (ndims=%d)\n", ndims);
    #endif

    return 0;
  }

  void cleanup() {
    if (allocated) {
      GPUFree(dim_gpu);
      GPUFree(stride_gpu);
      GPUFree(bounds_gpu);
      dim_gpu = NULL;
      stride_gpu = NULL;
      bounds_gpu = NULL;
      allocated = false;
      dim_stride_cached = false;
      cached_dim[0] = cached_dim[1] = cached_dim[2] = 0;
      cached_stride[0] = cached_stride[1] = cached_stride[2] = 0;
    }
  }

  ~GPUInterpMetadata() { cleanup(); }
};

/* Thread-local cache for metadata (avoids repeated allocations) */
static thread_local GPUInterpMetadata cached_metadata;

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

/* ==========================================================================
   SPECIALIZED MUSCL3 CHARACTERISTIC KERNELS FOR nvars=5 AND nvars=12
   These kernels use compile-time constants and fully unrolled operations
   for optimal register usage and instruction-level parallelism.
   ========================================================================== */

/* Specialized MUSCL3 kernel for nvars=5 (3D NavierStokes, no passive scalars) */
GPU_KERNEL void gpu_muscl3_interpolation_nd_char_ns3d_nvars5_kernel(
  double *fI, const double *fC, const double *u,
  int dim0, int dim1, int dim2,
  int stride0, int stride1, int stride2,
  int bounds0, int bounds1, int bounds2,
  int ghosts, int dir, int upw,
  double eps, double gamma
)
{
  const int total_interfaces = bounds0 * bounds1 * bounds2;
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  /* Decode tid to 3D interface index */
  const int i0 = tid % bounds0;
  const int i1 = (tid / bounds0) % bounds1;
  const int i2 = tid / (bounds0 * bounds1);

  /* Compute stride in current direction */
  const int stride_dir = (dir == 0) ? stride0 : ((dir == 1) ? stride1 : stride2);

  /* Constants */
  const double one_third = 1.0/3.0;
  const double one_sixth = 1.0/6.0;
  const int nvars = 5;

  /* Compute cell indices */
  int c0, c1, c2;
  int qm1, qm2, qp1, qp2;

  if (upw > 0) {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0 - 1;
    else if (dir == 1) c1 = i1 - 1;
    else c2 = i2 - 1;

    qm1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
  } else {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0;
    else if (dir == 1) c1 = i1;
    else c2 = i2;

    qp1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm1 = qp1 - stride_dir;
    qp2 = qp1 + stride_dir;
  }

  /* Load solution states for Roe average */
  double uL[5], uR[5];
  #pragma unroll
  for (int v = 0; v < 5; v++) {
    uL[v] = u[qm1*nvars + v];
    uR[v] = u[qp1*nvars + v];
  }

  /* Compute Roe average inline */
  double rhoL = uL[0], rhoR = uR[0];
  double tL = sqrt(rhoL), tR = sqrt(rhoR);
  double tLpR = tL + tR;

  double vxL = uL[1]/rhoL, vyL = uL[2]/rhoL, vzL = uL[3]/rhoL;
  double vxR = uR[1]/rhoR, vyR = uR[2]/rhoR, vzR = uR[3]/rhoR;

  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  double PL = (gamma-1.0) * (uL[4] - 0.5*rhoL*vsqL);
  double PR = (gamma-1.0) * (uR[4] - 0.5*rhoR*vsqR);
  double HL = 0.5*vsqL + gamma*PL/((gamma-1.0)*rhoL);
  double HR = 0.5*vsqR + gamma*PR/((gamma-1.0)*rhoR);

  /* rho = tL * tR is the Roe-averaged density, not needed for eigenvector formulation */
  double vx = (tL*vxL + tR*vxR) / tLpR;
  double vy = (tL*vyL + tR*vyR) / tLpR;
  double vz = (tL*vzL + tR*vzR) / tLpR;
  double H = (tL*HL + tR*HR) / tLpR;
  double vsq = vx*vx + vy*vy + vz*vz;
  double a2 = (gamma-1.0) * (H - 0.5*vsq);
  double a = sqrt(a2);

  /* Compute eigenvector coefficients */
  double gm1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double a2inv = 1.0 / a2;
  double twoA2inv = 0.5 * a2inv;
  double h0 = a2/gm1 + ek;

  /* Build left eigenvector matrix L (5x5) inline based on direction */
  double L[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) L[i] = 0.0;

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
  }

  /* Build right eigenvector matrix R (5x5) inline */
  double R[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) R[i] = 0.0;

  if (dir == 0) { /* X-direction */
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[0*5+1] = 1.0; R[1*5+1] = vx-a; R[2*5+1] = vy; R[3*5+1] = vz; R[4*5+1] = h0-a*vx;
    R[2*5+2] = -1.0; R[4*5+2] = -vy;
    R[3*5+3] = 1.0; R[4*5+3] = vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx+a; R[2*5+4] = vy; R[3*5+4] = vz; R[4*5+4] = h0+a*vx;
  } else if (dir == 1) { /* Y-direction */
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = 1.0; R[4*5+1] = vx;
    R[0*5+2] = 1.0; R[1*5+2] = vx; R[2*5+2] = vy-a; R[3*5+2] = vz; R[4*5+2] = h0-a*vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy+a; R[3*5+4] = vz; R[4*5+4] = h0+a*vy;
  } else { /* Z-direction */
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = -1.0; R[4*5+1] = -vx;
    R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[0*5+3] = 1.0; R[1*5+3] = vx; R[2*5+3] = vy; R[3*5+3] = vz-a; R[4*5+3] = h0-a*vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy; R[3*5+4] = vz+a; R[4*5+4] = h0+a*vz;
  }

  /* Transform to characteristic space, apply limiter, transform back */
  double fchar[5];

  #pragma unroll
  for (int v = 0; v < 5; v++) {
    double a_m2 = 0.0, a_m1 = 0.0, a_p1 = 0.0, a_p2 = 0.0;

    if (upw > 0) {
      #pragma unroll
      for (int k = 0; k < 5; k++) {
        double Lvk = L[v*5 + k];
        a_m2 += Lvk * fC[qm2*nvars + k];
        a_m1 += Lvk * fC[qm1*nvars + k];
        a_p1 += Lvk * fC[qp1*nvars + k];
      }
      double fdiff = a_p1 - a_m1;
      double bdiff = a_m1 - a_m2;
      double num = 3.0*fdiff*bdiff + eps;
      double den = 2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps;
      double limit = (den != 0.0) ? (num/den) : 1.0;
      fchar[v] = a_m1 + limit * (one_third*fdiff + one_sixth*bdiff);
    } else {
      #pragma unroll
      for (int k = 0; k < 5; k++) {
        double Lvk = L[v*5 + k];
        a_m1 += Lvk * fC[qm1*nvars + k];
        a_p1 += Lvk * fC[qp1*nvars + k];
        a_p2 += Lvk * fC[qp2*nvars + k];
      }
      double fdiff = a_p2 - a_p1;
      double bdiff = a_p1 - a_m1;
      double num = 3.0*fdiff*bdiff + eps;
      double den = 2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps;
      double limit = (den != 0.0) ? (num/den) : 1.0;
      fchar[v] = a_p1 - limit * (one_third*fdiff + one_sixth*bdiff);
    }
  }

  /* Transform back to physical space */
  #pragma unroll
  for (int k = 0; k < 5; k++) {
    double s = 0.0;
    #pragma unroll
    for (int v = 0; v < 5; v++) {
      s += R[k*5 + v] * fchar[v];
    }
    fI[tid*nvars + k] = s;
  }
}

/* Specialized MUSCL3 kernel for nvars=12 (3D NavierStokes + 7 passive scalars) */
GPU_KERNEL void gpu_muscl3_interpolation_nd_char_ns3d_nvars12_kernel(
  double *fI, const double *fC, const double *u,
  int dim0, int dim1, int dim2,
  int stride0, int stride1, int stride2,
  int bounds0, int bounds1, int bounds2,
  int ghosts, int dir, int upw,
  double eps, double gamma
)
{
  const int total_interfaces = bounds0 * bounds1 * bounds2;
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  /* Decode tid to 3D interface index */
  const int i0 = tid % bounds0;
  const int i1 = (tid / bounds0) % bounds1;
  const int i2 = tid / (bounds0 * bounds1);

  /* Compute stride in current direction */
  const int stride_dir = (dir == 0) ? stride0 : ((dir == 1) ? stride1 : stride2);

  /* Constants */
  const double one_third = 1.0/3.0;
  const double one_sixth = 1.0/6.0;
  const int nvars = 12;
  const int base_nvars = 5;

  /* Compute cell indices */
  int c0, c1, c2;
  int qm1, qm2, qp1, qp2;

  if (upw > 0) {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0 - 1;
    else if (dir == 1) c1 = i1 - 1;
    else c2 = i2 - 1;

    qm1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
  } else {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0;
    else if (dir == 1) c1 = i1;
    else c2 = i2;

    qp1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm1 = qp1 - stride_dir;
    qp2 = qp1 + stride_dir;
  }

  /* Load solution states for Roe average (only base 5 vars needed) */
  double uL[5], uR[5];
  #pragma unroll
  for (int v = 0; v < 5; v++) {
    uL[v] = u[qm1*nvars + v];
    uR[v] = u[qp1*nvars + v];
  }

  /* Compute Roe average inline */
  double rhoL = uL[0], rhoR = uR[0];
  double tL = sqrt(rhoL), tR = sqrt(rhoR);
  double tLpR = tL + tR;

  double vxL = uL[1]/rhoL, vyL = uL[2]/rhoL, vzL = uL[3]/rhoL;
  double vxR = uR[1]/rhoR, vyR = uR[2]/rhoR, vzR = uR[3]/rhoR;

  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  double PL = (gamma-1.0) * (uL[4] - 0.5*rhoL*vsqL);
  double PR = (gamma-1.0) * (uR[4] - 0.5*rhoR*vsqR);
  double HL = 0.5*vsqL + gamma*PL/((gamma-1.0)*rhoL);
  double HR = 0.5*vsqR + gamma*PR/((gamma-1.0)*rhoR);

  double vx = (tL*vxL + tR*vxR) / tLpR;
  double vy = (tL*vyL + tR*vyR) / tLpR;
  double vz = (tL*vzL + tR*vzR) / tLpR;
  double H = (tL*HL + tR*HR) / tLpR;
  double vsq = vx*vx + vy*vy + vz*vz;
  double a2 = (gamma-1.0) * (H - 0.5*vsq);
  double a = sqrt(a2);

  /* Compute eigenvector coefficients */
  double gm1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double a2inv = 1.0 / a2;
  double twoA2inv = 0.5 * a2inv;
  double h0 = a2/gm1 + ek;

  /* Build left eigenvector matrix L (5x5) for base NS3D system */
  double L[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) L[i] = 0.0;

  if (dir == 0) {
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
  } else if (dir == 1) {
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
  } else {
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
  }

  /* Build right eigenvector matrix R (5x5) */
  double R[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) R[i] = 0.0;

  if (dir == 0) {
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[0*5+1] = 1.0; R[1*5+1] = vx-a; R[2*5+1] = vy; R[3*5+1] = vz; R[4*5+1] = h0-a*vx;
    R[2*5+2] = -1.0; R[4*5+2] = -vy;
    R[3*5+3] = 1.0; R[4*5+3] = vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx+a; R[2*5+4] = vy; R[3*5+4] = vz; R[4*5+4] = h0+a*vx;
  } else if (dir == 1) {
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = 1.0; R[4*5+1] = vx;
    R[0*5+2] = 1.0; R[1*5+2] = vx; R[2*5+2] = vy-a; R[3*5+2] = vz; R[4*5+2] = h0-a*vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy+a; R[3*5+4] = vz; R[4*5+4] = h0+a*vy;
  } else {
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = -1.0; R[4*5+1] = -vx;
    R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[0*5+3] = 1.0; R[1*5+3] = vx; R[2*5+3] = vy; R[3*5+3] = vz-a; R[4*5+3] = h0-a*vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy; R[3*5+4] = vz+a; R[4*5+4] = h0+a*vz;
  }

  /* Process base 5 NS3D variables with characteristic transform */
  double fchar[5];

  #pragma unroll
  for (int v = 0; v < base_nvars; v++) {
    double a_m2 = 0.0, a_m1 = 0.0, a_p1 = 0.0, a_p2 = 0.0;

    if (upw > 0) {
      #pragma unroll
      for (int k = 0; k < base_nvars; k++) {
        double Lvk = L[v*5 + k];
        a_m2 += Lvk * fC[qm2*nvars + k];
        a_m1 += Lvk * fC[qm1*nvars + k];
        a_p1 += Lvk * fC[qp1*nvars + k];
      }
      double fdiff = a_p1 - a_m1;
      double bdiff = a_m1 - a_m2;
      double num = 3.0*fdiff*bdiff + eps;
      double den = 2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps;
      double limit = (den != 0.0) ? (num/den) : 1.0;
      fchar[v] = a_m1 + limit * (one_third*fdiff + one_sixth*bdiff);
    } else {
      #pragma unroll
      for (int k = 0; k < base_nvars; k++) {
        double Lvk = L[v*5 + k];
        a_m1 += Lvk * fC[qm1*nvars + k];
        a_p1 += Lvk * fC[qp1*nvars + k];
        a_p2 += Lvk * fC[qp2*nvars + k];
      }
      double fdiff = a_p2 - a_p1;
      double bdiff = a_p1 - a_m1;
      double num = 3.0*fdiff*bdiff + eps;
      double den = 2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps;
      double limit = (den != 0.0) ? (num/den) : 1.0;
      fchar[v] = a_p1 - limit * (one_third*fdiff + one_sixth*bdiff);
    }
  }

  /* Transform base variables back to physical space */
  #pragma unroll
  for (int k = 0; k < base_nvars; k++) {
    double s = 0.0;
    #pragma unroll
    for (int v = 0; v < base_nvars; v++) {
      s += R[k*5 + v] * fchar[v];
    }
    fI[tid*nvars + k] = s;
  }

  /* Process passive scalars (vars 5-11) with component-wise MUSCL3 */
  #pragma unroll
  for (int k = base_nvars; k < nvars; k++) {
    if (upw > 0) {
      double m2 = fC[qm2*nvars + k];
      double m1 = fC[qm1*nvars + k];
      double p1 = fC[qp1*nvars + k];
      double fdiff = p1 - m1;
      double bdiff = m1 - m2;
      double num = 3.0*fdiff*bdiff + eps;
      double den = 2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps;
      double limit = (den != 0.0) ? (num/den) : 1.0;
      fI[tid*nvars + k] = m1 + limit * (one_third*fdiff + one_sixth*bdiff);
    } else {
      double m1 = fC[qm1*nvars + k];
      double p1 = fC[qp1*nvars + k];
      double p2 = fC[qp2*nvars + k];
      double fdiff = p2 - p1;
      double bdiff = p1 - m1;
      double num = 3.0*fdiff*bdiff + eps;
      double den = 2.0*(fdiff-bdiff)*(fdiff-bdiff) + 3.0*fdiff*bdiff + eps;
      double limit = (den != 0.0) ? (num/den) : 1.0;
      fI[tid*nvars + k] = p1 - limit * (one_third*fdiff + one_sixth*bdiff);
    }
  }
}

/* ==========================================================================
   FUSED WENO5 CHARACTERISTIC KERNELS FOR nvars=5 AND nvars=12
   These kernels compute WENO weights AND interpolation in a SINGLE pass,
   eliminating intermediate weight storage and duplicate Roe average computation.
   ========================================================================== */

/* Helper: Compute WENO5 smoothness indicators inline */
static __device__ __forceinline__ void weno5_smoothness_inline(
  double m3, double m2, double m1, double p1, double p2,
  double *b1, double *b2, double *b3
)
{
  const double c13_12 = 13.0/12.0;
  const double c1_4 = 0.25;

  *b1 = c13_12*(m3 - 2*m2 + m1)*(m3 - 2*m2 + m1)
      + c1_4*(m3 - 4*m2 + 3*m1)*(m3 - 4*m2 + 3*m1);
  *b2 = c13_12*(m2 - 2*m1 + p1)*(m2 - 2*m1 + p1)
      + c1_4*(m2 - p1)*(m2 - p1);
  *b3 = c13_12*(m1 - 2*p1 + p2)*(m1 - 2*p1 + p2)
      + c1_4*(3*m1 - 4*p1 + p2)*(3*m1 - 4*p1 + p2);
}

/* Helper: Compute WENO5 weights and interpolated value inline (Jiang-Shu) */
static __device__ __forceinline__ double weno5_interp_inline(
  double m3, double m2, double m1, double p1, double p2, double eps
)
{
  /* Smoothness indicators */
  double b1, b2, b3;
  weno5_smoothness_inline(m3, m2, m1, p1, p2, &b1, &b2, &b3);

  /* Optimal weights */
  const double c1 = 0.1, c2 = 0.6, c3 = 0.3;

  /* Nonlinear weights (Jiang-Shu) */
  double a1 = c1 / ((b1 + eps) * (b1 + eps));
  double a2 = c2 / ((b2 + eps) * (b2 + eps));
  double a3 = c3 / ((b3 + eps) * (b3 + eps));
  double asum_inv = 1.0 / (a1 + a2 + a3);
  double w1 = a1 * asum_inv;
  double w2 = a2 * asum_inv;
  double w3 = a3 * asum_inv;

  /* Candidate stencils */
  const double c1_6 = 1.0/6.0;
  double f1 = (2*c1_6)*m3 + (-7*c1_6)*m2 + (11*c1_6)*m1;
  double f2 = (-c1_6)*m2 + (5*c1_6)*m1 + (2*c1_6)*p1;
  double f3 = (2*c1_6)*m1 + (5*c1_6)*p1 + (-c1_6)*p2;

  return w1*f1 + w2*f2 + w3*f3;
}

/* Fused WENO5 kernel for nvars=5 (3D NavierStokes, no passive scalars)
   Computes weights and interpolation in a single pass */
GPU_KERNEL void gpu_weno5_fused_char_ns3d_nvars5_kernel(
  double *fI, const double *fC, const double *u,
  int dim0, int dim1, int dim2,
  int stride0, int stride1, int stride2,
  int bounds0, int bounds1, int bounds2,
  int ghosts, int dir, int upw,
  double eps, double gamma
)
{
  const int total_interfaces = bounds0 * bounds1 * bounds2;
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  /* Decode tid to 3D interface index */
  const int i0 = tid % bounds0;
  const int i1 = (tid / bounds0) % bounds1;
  const int i2 = tid / (bounds0 * bounds1);

  const int stride_dir = (dir == 0) ? stride0 : ((dir == 1) ? stride1 : stride2);
  const int nvars = 5;

  /* Compute cell indices */
  int c0, c1, c2;
  int qm1, qm2, qm3, qp1, qp2, qL, qR;

  if (upw > 0) {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0 - 1;
    else if (dir == 1) c1 = i1 - 1;
    else c2 = i2 - 1;

    qm1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm3 = qm1 - 2*stride_dir;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
    qp2 = qm1 + 2*stride_dir;
    qL = qm1;
    qR = qp1;
  } else {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0;
    else if (dir == 1) c1 = i1;
    else c2 = i2;

    qm1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm3 = qm1 + 2*stride_dir;
    qm2 = qm1 + stride_dir;
    qp1 = qm1 - stride_dir;
    qp2 = qm1 - 2*stride_dir;
    qL = qm1;
    qR = qp1;
  }

  /* Load solution states for Roe average */
  double uL[5], uR[5];
  #pragma unroll
  for (int v = 0; v < 5; v++) {
    uL[v] = u[qL*nvars + v];
    uR[v] = u[qR*nvars + v];
  }

  /* Compute Roe average */
  double rhoL = uL[0], rhoR = uR[0];
  double tL = sqrt(rhoL), tR = sqrt(rhoR);
  double tLpR = tL + tR;

  double vxL = uL[1]/rhoL, vyL = uL[2]/rhoL, vzL = uL[3]/rhoL;
  double vxR = uR[1]/rhoR, vyR = uR[2]/rhoR, vzR = uR[3]/rhoR;

  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  double PL = (gamma-1.0) * (uL[4] - 0.5*rhoL*vsqL);
  double PR = (gamma-1.0) * (uR[4] - 0.5*rhoR*vsqR);
  double HL = 0.5*vsqL + gamma*PL/((gamma-1.0)*rhoL);
  double HR = 0.5*vsqR + gamma*PR/((gamma-1.0)*rhoR);

  double vx = (tL*vxL + tR*vxR) / tLpR;
  double vy = (tL*vyL + tR*vyR) / tLpR;
  double vz = (tL*vzL + tR*vzR) / tLpR;
  double H = (tL*HL + tR*HR) / tLpR;
  double vsq = vx*vx + vy*vy + vz*vz;
  double a2 = (gamma-1.0) * (H - 0.5*vsq);
  double a = sqrt(a2);

  /* Eigenvector coefficients */
  double gm1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double a2inv = 1.0 / a2;
  double twoA2inv = 0.5 * a2inv;
  double h0 = a2/gm1 + ek;

  /* Build left eigenvector matrix L (5x5) */
  double L[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) L[i] = 0.0;

  if (dir == 0) {
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
  } else if (dir == 1) {
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
  } else {
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
  }

  /* Build right eigenvector matrix R (5x5) */
  double R[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) R[i] = 0.0;

  if (dir == 0) {
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[0*5+1] = 1.0; R[1*5+1] = vx-a; R[2*5+1] = vy; R[3*5+1] = vz; R[4*5+1] = h0-a*vx;
    R[2*5+2] = -1.0; R[4*5+2] = -vy;
    R[3*5+3] = 1.0; R[4*5+3] = vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx+a; R[2*5+4] = vy; R[3*5+4] = vz; R[4*5+4] = h0+a*vx;
  } else if (dir == 1) {
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = 1.0; R[4*5+1] = vx;
    R[0*5+2] = 1.0; R[1*5+2] = vx; R[2*5+2] = vy-a; R[3*5+2] = vz; R[4*5+2] = h0-a*vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy+a; R[3*5+4] = vz; R[4*5+4] = h0+a*vy;
  } else {
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = -1.0; R[4*5+1] = -vx;
    R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[0*5+3] = 1.0; R[1*5+3] = vx; R[2*5+3] = vy; R[3*5+3] = vz-a; R[4*5+3] = h0-a*vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy; R[3*5+4] = vz+a; R[4*5+4] = h0+a*vz;
  }

  /* Transform to characteristic, apply WENO5, transform back */
  double fchar[5];

  #pragma unroll
  for (int v = 0; v < 5; v++) {
    /* Transform stencil to characteristic space */
    double fm3 = 0.0, fm2 = 0.0, fm1 = 0.0, fp1 = 0.0, fp2 = 0.0;
    #pragma unroll
    for (int k = 0; k < 5; k++) {
      double Lvk = L[v*5 + k];
      fm3 += Lvk * fC[qm3*nvars + k];
      fm2 += Lvk * fC[qm2*nvars + k];
      fm1 += Lvk * fC[qm1*nvars + k];
      fp1 += Lvk * fC[qp1*nvars + k];
      fp2 += Lvk * fC[qp2*nvars + k];
    }

    /* Apply WENO5 inline */
    fchar[v] = weno5_interp_inline(fm3, fm2, fm1, fp1, fp2, eps);
  }

  /* Transform back to physical space */
  #pragma unroll
  for (int k = 0; k < 5; k++) {
    double s = 0.0;
    #pragma unroll
    for (int v = 0; v < 5; v++) {
      s += R[k*5 + v] * fchar[v];
    }
    fI[tid*nvars + k] = s;
  }
}

/* Fused WENO5 kernel for nvars=12 (3D NavierStokes + 7 passive scalars) */
GPU_KERNEL void gpu_weno5_fused_char_ns3d_nvars12_kernel(
  double *fI, const double *fC, const double *u,
  int dim0, int dim1, int dim2,
  int stride0, int stride1, int stride2,
  int bounds0, int bounds1, int bounds2,
  int ghosts, int dir, int upw,
  double eps, double gamma
)
{
  const int total_interfaces = bounds0 * bounds1 * bounds2;
  const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= total_interfaces) return;

  const int i0 = tid % bounds0;
  const int i1 = (tid / bounds0) % bounds1;
  const int i2 = tid / (bounds0 * bounds1);

  const int stride_dir = (dir == 0) ? stride0 : ((dir == 1) ? stride1 : stride2);
  const int nvars = 12;
  const int base_nvars = 5;

  int c0, c1, c2;
  int qm1, qm2, qm3, qp1, qp2, qL, qR;

  if (upw > 0) {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0 - 1;
    else if (dir == 1) c1 = i1 - 1;
    else c2 = i2 - 1;

    qm1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm3 = qm1 - 2*stride_dir;
    qm2 = qm1 - stride_dir;
    qp1 = qm1 + stride_dir;
    qp2 = qm1 + 2*stride_dir;
    qL = qm1;
    qR = qp1;
  } else {
    c0 = i0; c1 = i1; c2 = i2;
    if (dir == 0) c0 = i0;
    else if (dir == 1) c1 = i1;
    else c2 = i2;

    qm1 = (c2 + ghosts) * stride2 + (c1 + ghosts) * stride1 + (c0 + ghosts) * stride0;
    qm3 = qm1 + 2*stride_dir;
    qm2 = qm1 + stride_dir;
    qp1 = qm1 - stride_dir;
    qp2 = qm1 - 2*stride_dir;
    qL = qm1;
    qR = qp1;
  }

  /* Load solution states for Roe average */
  double uL[5], uR[5];
  #pragma unroll
  for (int v = 0; v < 5; v++) {
    uL[v] = u[qL*nvars + v];
    uR[v] = u[qR*nvars + v];
  }

  /* Compute Roe average */
  double rhoL = uL[0], rhoR = uR[0];
  double tL = sqrt(rhoL), tR = sqrt(rhoR);
  double tLpR = tL + tR;

  double vxL = uL[1]/rhoL, vyL = uL[2]/rhoL, vzL = uL[3]/rhoL;
  double vxR = uR[1]/rhoR, vyR = uR[2]/rhoR, vzR = uR[3]/rhoR;

  double vsqL = vxL*vxL + vyL*vyL + vzL*vzL;
  double vsqR = vxR*vxR + vyR*vyR + vzR*vzR;
  double PL = (gamma-1.0) * (uL[4] - 0.5*rhoL*vsqL);
  double PR = (gamma-1.0) * (uR[4] - 0.5*rhoR*vsqR);
  double HL = 0.5*vsqL + gamma*PL/((gamma-1.0)*rhoL);
  double HR = 0.5*vsqR + gamma*PR/((gamma-1.0)*rhoR);

  double vx = (tL*vxL + tR*vxR) / tLpR;
  double vy = (tL*vyL + tR*vyR) / tLpR;
  double vz = (tL*vzL + tR*vzR) / tLpR;
  double H = (tL*HL + tR*HR) / tLpR;
  double vsq = vx*vx + vy*vy + vz*vz;
  double a2 = (gamma-1.0) * (H - 0.5*vsq);
  double a = sqrt(a2);

  double gm1 = gamma - 1.0;
  double ek = 0.5 * vsq;
  double a2inv = 1.0 / a2;
  double twoA2inv = 0.5 * a2inv;
  double h0 = a2/gm1 + ek;

  /* Build L and R matrices (5x5) */
  double L[25], R[25];
  #pragma unroll
  for (int i = 0; i < 25; i++) { L[i] = 0.0; R[i] = 0.0; }

  if (dir == 0) {
    L[5*1+0] = (gm1*ek + a*vx) * twoA2inv; L[5*1+1] = (-gm1*vx - a) * twoA2inv;
    L[5*1+2] = (-gm1*vy) * twoA2inv; L[5*1+3] = (-gm1*vz) * twoA2inv; L[5*1+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv; L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv; L[5*0+3] = gm1*vz * a2inv; L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vx) * twoA2inv; L[5*4+1] = (-gm1*vx + a) * twoA2inv;
    L[5*4+2] = (-gm1*vy) * twoA2inv; L[5*4+3] = (-gm1*vz) * twoA2inv; L[5*4+4] = gm1 * twoA2inv;
    L[5*2+0] = vy; L[5*2+2] = -1.0; L[5*3+0] = -vz; L[5*3+3] = 1.0;
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[0*5+1] = 1.0; R[1*5+1] = vx-a; R[2*5+1] = vy; R[3*5+1] = vz; R[4*5+1] = h0-a*vx;
    R[2*5+2] = -1.0; R[4*5+2] = -vy; R[3*5+3] = 1.0; R[4*5+3] = vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx+a; R[2*5+4] = vy; R[3*5+4] = vz; R[4*5+4] = h0+a*vx;
  } else if (dir == 1) {
    L[5*2+0] = (gm1*ek + a*vy) * twoA2inv; L[5*2+1] = (-gm1*vx) * twoA2inv;
    L[5*2+2] = (-gm1*vy - a) * twoA2inv; L[5*2+3] = (-gm1*vz) * twoA2inv; L[5*2+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv; L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv; L[5*0+3] = gm1*vz * a2inv; L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vy) * twoA2inv; L[5*4+1] = (-gm1*vx) * twoA2inv;
    L[5*4+2] = (-gm1*vy + a) * twoA2inv; L[5*4+3] = (-gm1*vz) * twoA2inv; L[5*4+4] = gm1 * twoA2inv;
    L[5*1+0] = -vx; L[5*1+1] = 1.0; L[5*3+0] = vz; L[5*3+3] = -1.0;
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = 1.0; R[4*5+1] = vx;
    R[0*5+2] = 1.0; R[1*5+2] = vx; R[2*5+2] = vy-a; R[3*5+2] = vz; R[4*5+2] = h0-a*vy;
    R[3*5+3] = -1.0; R[4*5+3] = -vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy+a; R[3*5+4] = vz; R[4*5+4] = h0+a*vy;
  } else {
    L[5*3+0] = (gm1*ek + a*vz) * twoA2inv; L[5*3+1] = (-gm1*vx) * twoA2inv;
    L[5*3+2] = (-gm1*vy) * twoA2inv; L[5*3+3] = (-gm1*vz - a) * twoA2inv; L[5*3+4] = gm1 * twoA2inv;
    L[5*0+0] = 1.0 - gm1*ek * a2inv; L[5*0+1] = gm1*vx * a2inv;
    L[5*0+2] = gm1*vy * a2inv; L[5*0+3] = gm1*vz * a2inv; L[5*0+4] = -gm1 * a2inv;
    L[5*4+0] = (gm1*ek - a*vz) * twoA2inv; L[5*4+1] = (-gm1*vx) * twoA2inv;
    L[5*4+2] = (-gm1*vy) * twoA2inv; L[5*4+3] = (-gm1*vz + a) * twoA2inv; L[5*4+4] = gm1 * twoA2inv;
    L[5*1+0] = vx; L[5*1+1] = -1.0; L[5*2+0] = -vy; L[5*2+2] = 1.0;
    R[0*5+0] = 1.0; R[1*5+0] = vx; R[2*5+0] = vy; R[3*5+0] = vz; R[4*5+0] = ek;
    R[1*5+1] = -1.0; R[4*5+1] = -vx; R[2*5+2] = 1.0; R[4*5+2] = vy;
    R[0*5+3] = 1.0; R[1*5+3] = vx; R[2*5+3] = vy; R[3*5+3] = vz-a; R[4*5+3] = h0-a*vz;
    R[0*5+4] = 1.0; R[1*5+4] = vx; R[2*5+4] = vy; R[3*5+4] = vz+a; R[4*5+4] = h0+a*vz;
  }

  /* Process base 5 NS3D variables with characteristic WENO5 */
  double fchar[5];

  #pragma unroll
  for (int v = 0; v < base_nvars; v++) {
    double fm3 = 0.0, fm2 = 0.0, fm1 = 0.0, fp1 = 0.0, fp2 = 0.0;
    #pragma unroll
    for (int k = 0; k < base_nvars; k++) {
      double Lvk = L[v*5 + k];
      fm3 += Lvk * fC[qm3*nvars + k];
      fm2 += Lvk * fC[qm2*nvars + k];
      fm1 += Lvk * fC[qm1*nvars + k];
      fp1 += Lvk * fC[qp1*nvars + k];
      fp2 += Lvk * fC[qp2*nvars + k];
    }
    fchar[v] = weno5_interp_inline(fm3, fm2, fm1, fp1, fp2, eps);
  }

  /* Transform back to physical space */
  #pragma unroll
  for (int k = 0; k < base_nvars; k++) {
    double s = 0.0;
    #pragma unroll
    for (int v = 0; v < base_nvars; v++) {
      s += R[k*5 + v] * fchar[v];
    }
    fI[tid*nvars + k] = s;
  }

  /* Passive scalars: component-wise WENO5 */
  #pragma unroll
  for (int k = base_nvars; k < nvars; k++) {
    double m3 = fC[qm3*nvars + k];
    double m2 = fC[qm2*nvars + k];
    double m1 = fC[qm1*nvars + k];
    double p1 = fC[qp1*nvars + k];
    double p2 = fC[qp2*nvars + k];
    fI[tid*nvars + k] = weno5_interp_inline(m3, m2, m1, p1, p2, eps);
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);

  /* Use cached metadata to avoid repeated alloc/copy/free */
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");
    return;
  }

  /* Compute total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int gridSize = (total_interfaces + blockSize - 1) / blockSize;

  GPU_KERNEL_LAUNCH(gpu_weno5_interpolation_nd_kernel, gridSize, blockSize)(
    fI, fC, w1, w2, w3, nvars, ndims, cached_metadata.dim_gpu, cached_metadata.stride_gpu,
    cached_metadata.bounds_gpu, ghosts, dir, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());

  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);

  /* Use cached metadata to avoid repeated alloc/copy/free */
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for characteristic interpolation\n");
    return;
  }

  /* Compute total number of interface points */
  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) {
    total_interfaces *= bounds_inter[i];
  }

  int gridSize = (total_interfaces + blockSize - 1) / blockSize;

  GPU_KERNEL_LAUNCH(gpu_weno5_interpolation_nd_char_kernel, gridSize, blockSize)(
    fI, fC, u, w1, w2, w3, nvars, ndims, cached_metadata.dim_gpu, cached_metadata.stride_gpu,
    cached_metadata.bounds_gpu, ghosts, dir, upw, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());

  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\\n");
    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl2_interpolation_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, limiter_id
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, eps
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_muscl2_interpolation_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, limiter_id, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;

  /* Use specialized kernels for 3D with nvars=5 or nvars=12 */
  if (ndims == 3 && nvars == 5) {
    GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_nd_char_ns3d_nvars5_kernel, gridSize, blockSize)(
      fI, fC, u,
      dim[0], dim[1], dim[2],
      stride_with_ghosts[0], stride_with_ghosts[1], stride_with_ghosts[2],
      bounds_inter[0], bounds_inter[1], bounds_inter[2],
      ghosts, dir, upw, eps, gamma
    );
  } else if (ndims == 3 && nvars == 12) {
    GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_nd_char_ns3d_nvars12_kernel, gridSize, blockSize)(
      fI, fC, u,
      dim[0], dim[1], dim[2],
      stride_with_ghosts[0], stride_with_ghosts[1], stride_with_ghosts[2],
      bounds_inter[0], bounds_inter[1], bounds_inter[2],
      ghosts, dir, upw, eps, gamma
    );
  } else {
    /* Fall back to generic kernel */
    if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
      fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");
      return;
    }
    int *dim_gpu = cached_metadata.dim_gpu;
    int *stride_gpu = cached_metadata.stride_gpu;
    int *bounds_gpu = cached_metadata.bounds_gpu;

    GPU_KERNEL_LAUNCH(gpu_muscl3_interpolation_nd_char_ns3d_kernel, gridSize, blockSize)(
      fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, eps, gamma
    );
  }
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_order_upwind_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_second_order_central_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fourth_order_central_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fifth_order_upwind_nd_kernel, gridSize, blockSize)(
    fI, fC, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_first_order_upwind_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_second_order_central_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fourth_order_central_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
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
  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);
  if (cached_metadata.setup(dim, stride_with_ghosts, bounds_inter, ndims)) {
    fprintf(stderr, "Error: Failed to setup GPU metadata for interpolation\n");

    return;
  }
  int *dim_gpu = cached_metadata.dim_gpu;
  int *stride_gpu = cached_metadata.stride_gpu;
  int *bounds_gpu = cached_metadata.bounds_gpu;

  int total_interfaces = 1;
  for (int i = 0; i < ndims; i++) total_interfaces *= bounds_inter[i];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_fifth_order_upwind_nd_char_ns3d_kernel, gridSize, blockSize)(
    fI, fC, u, nvars, ndims, dim_gpu, stride_gpu, bounds_gpu, ghosts, dir, upw, gamma
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  /* No need to free - metadata cached for reuse */
#endif
}

/* Launch wrapper for FUSED WENO5 characteristic interpolation (NS3D) */
/* This function computes weights and interpolation in a single pass */
/* Only works for 3D with nvars=5 or nvars=12; returns 0 on success, -1 if not applicable */
extern "C" int gpu_launch_weno5_fused_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps, double gamma,
  int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fC; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts;
  (void)bounds_inter; (void)ghosts; (void)dir; (void)upw; (void)eps; (void)gamma; (void)blockSize;
  return -1;
#else
  /* Only support 3D with nvars=5 or nvars=12 */
  if (ndims != 3 || (nvars != 5 && nvars != 12)) {
    return -1;
  }

  if (blockSize <= 0) blockSize = GPUGetBlockSize("compute_bound", nvars);

  int total_interfaces = bounds_inter[0] * bounds_inter[1] * bounds_inter[2];
  int gridSize = (total_interfaces + blockSize - 1) / blockSize;

  if (nvars == 5) {
    GPU_KERNEL_LAUNCH(gpu_weno5_fused_char_ns3d_nvars5_kernel, gridSize, blockSize)(
      fI, fC, u,
      dim[0], dim[1], dim[2],
      stride_with_ghosts[0], stride_with_ghosts[1], stride_with_ghosts[2],
      bounds_inter[0], bounds_inter[1], bounds_inter[2],
      ghosts, dir, upw, eps, gamma
    );
  } else {
    GPU_KERNEL_LAUNCH(gpu_weno5_fused_char_ns3d_nvars12_kernel, gridSize, blockSize)(
      fI, fC, u,
      dim[0], dim[1], dim[2],
      stride_with_ghosts[0], stride_with_ghosts[1], stride_with_ghosts[2],
      bounds_inter[0], bounds_inter[1], bounds_inter[2],
      ghosts, dir, upw, eps, gamma
    );
  }
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
  return 0;
#endif
}

