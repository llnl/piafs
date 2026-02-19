/*! @file gpu_derivative.cu
    @brief GPU kernels for first derivative computation
*/

#include <gpu.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: 2nd order central first derivative */
GPU_KERNEL void gpu_first_derivative_second_order_kernel(
  double *Df,           /* output: derivative */
  const double *f,      /* input: function values */
  int nvars,            /* number of variables */
  int npoints,          /* number of grid points */
  int ghosts,           /* number of ghost points */
  int stride            /* stride along the direction */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npoints) return;

  /* Compute derivatives at ALL points including ghosts, matching CPU behavior */
  int qC = idx * stride;  /* current point */
  int qL = (idx - 1) * stride;  /* left neighbor */
  int qR = (idx + 1) * stride;  /* right neighbor */

  for (int v = 0; v < nvars; v++) {
    if (idx == 0) {
      /* First point: one-sided forward */
      int qRR = (idx + 2) * stride;
      Df[qC*nvars+v] = 0.5 * (-3.0*f[qC*nvars+v] + 4.0*f[qR*nvars+v] - f[qRR*nvars+v]);
    } else if (idx == npoints - 1) {
      /* Last point: one-sided backward */
      int qLL = (idx - 2) * stride;
      Df[qC*nvars+v] = 0.5 * (3.0*f[qC*nvars+v] - 4.0*f[qL*nvars+v] + f[qLL*nvars+v]);
    } else {
      /* All other points: central difference */
      Df[qC*nvars+v] = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
    }
  }
}

/* Kernel: 4th order central first derivative */
GPU_KERNEL void gpu_first_derivative_fourth_order_kernel(
  double *Df,           /* output: derivative */
  const double *f,      /* input: function values */
  int nvars,            /* number of variables */
  int npoints,          /* number of grid points */
  int ghosts,           /* number of ghost points */
  int stride            /* stride along the direction */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npoints) return;

  const double one_twelve = 1.0/12.0;
  int qC = idx * stride;

  /* Match CPU: use different schemes depending on position */
  if (idx == 0) {
    /* First point: 4th order forward biased */
    int qp1 = (idx + 1) * stride;
    int qp2 = (idx + 2) * stride;
    int qp3 = (idx + 3) * stride;
    int qp4 = (idx + 4) * stride;

    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = one_twelve * (-25.0*f[qC*nvars+v] + 48.0*f[qp1*nvars+v] - 36.0*f[qp2*nvars+v] + 16.0*f[qp3*nvars+v] - 3.0*f[qp4*nvars+v]);
    }
  } else if (idx == 1) {
    /* Second point: 4th order forward biased */
    int qm1 = (idx - 1) * stride;
    int qp1 = (idx + 1) * stride;
    int qp2 = (idx + 2) * stride;
    int qp3 = (idx + 3) * stride;

    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = one_twelve * (-3.0*f[qm1*nvars+v] - 10.0*f[qC*nvars+v] + 18.0*f[qp1*nvars+v] - 6.0*f[qp2*nvars+v] + f[qp3*nvars+v]);
    }
  } else if (idx >= 2 && idx < npoints - 2) {
    /* Interior: 4th order central */
    int qL = (idx - 1) * stride;
    int qR = (idx + 1) * stride;
    int qLL = (idx - 2) * stride;
    int qRR = (idx + 2) * stride;

    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = one_twelve * (f[qLL*nvars+v] - 8.0*f[qL*nvars+v] + 8.0*f[qR*nvars+v] - f[qRR*nvars+v]);
    }
  } else if (idx == npoints - 2) {
    /* Second-to-last point: 4th order backward biased */
    int qm3 = (idx - 3) * stride;
    int qm2 = (idx - 2) * stride;
    int qm1 = (idx - 1) * stride;
    int qp1 = (idx + 1) * stride;

    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = one_twelve * (-f[qm3*nvars+v] + 6.0*f[qm2*nvars+v] - 18.0*f[qm1*nvars+v] + 10.0*f[qC*nvars+v] + 3.0*f[qp1*nvars+v]);
    }
  } else if (idx == npoints - 1) {
    /* Last point: 4th order backward biased */
    int qm4 = (idx - 4) * stride;
    int qm3 = (idx - 3) * stride;
    int qm2 = (idx - 2) * stride;
    int qm1 = (idx - 1) * stride;

    for (int v = 0; v < nvars; v++) {
      Df[qC*nvars+v] = one_twelve * (3.0*f[qm4*nvars+v] - 16.0*f[qm3*nvars+v] + 36.0*f[qm2*nvars+v] - 48.0*f[qm1*nvars+v] + 25.0*f[qC*nvars+v]);
    }
  }
}

/* Kernel: 1st order upwind first derivative */
GPU_KERNEL void gpu_first_derivative_first_order_kernel(
  double *Df,           /* output: derivative */
  const double *f,      /* input: function values */
  int nvars,            /* number of variables */
  int npoints,          /* number of grid points */
  int ghosts,           /* number of ghost points */
  int stride,           /* stride along the direction */
  double bias           /* bias: >0 forward, <0 backward, =0 central */
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < npoints) {
    int qC = idx * stride;
    int qL = (idx - 1) * stride;
    int qR = (idx + 1) * stride;

    for (int v = 0; v < nvars; v++) {
      if (idx == 0) {
        /* Left boundary ghost point: forward difference */
        Df[qC*nvars+v] = f[qR*nvars+v] - f[qC*nvars+v];
      } else if (idx == npoints - 1) {
        /* Right boundary ghost point: backward difference */
        Df[qC*nvars+v] = f[qC*nvars+v] - f[qL*nvars+v];
      } else {
        /* Interior and other ghost points: biased difference */
        Df[qC*nvars+v] = (bias > 0 ? f[qR*nvars+v] - f[qC*nvars+v] :
                         bias < 0 ? f[qC*nvars+v] - f[qL*nvars+v] :
                         0.5 * (f[qR*nvars+v] - f[qL*nvars+v]));
      }
    }
  }
}

/*******************************************************************************
 * 3D BATCHED DERIVATIVE KERNELS
 * These kernels process the entire 3D domain in a single launch, avoiding
 * the overhead of launching millions of tiny kernels per line.
 ******************************************************************************/

/* Helper device function to compute linear index from 3D coordinates */
__device__ __forceinline__ int idx3d(int i, int j, int k, int ni, int nj)
{
  return i + ni * (j + nj * k);
}

/* Kernel: 1st order first derivative for entire 3D domain
 * Each thread processes one grid point across all variables.
 * Only processes interior (i,k) positions to match per-line behavior.
 * dir: 0=X, 1=Y, 2=Z direction */
GPU_KERNEL void gpu_first_derivative_first_order_3d_kernel(
  double *Df,           /* output: derivative */
  const double *f,      /* input: function values */
  int nvars,            /* number of variables */
  int ni,               /* dimension including ghosts in X */
  int nj,               /* dimension including ghosts in Y */
  int nk,               /* dimension including ghosts in Z */
  int ghosts,           /* number of ghost points */
  int dir,              /* derivative direction: 0=X, 1=Y, 2=Z */
  double bias           /* bias: >0 forward, <0 backward, =0 central */
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  /* Convert linear index to 3D indices */
  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  /* Skip ghost regions in non-derivative directions to match per-line behavior */
  if (dir == 0) {
    if (j < ghosts || j >= nj - ghosts || k < ghosts || k >= nk - ghosts) return;
  } else if (dir == 1) {
    if (i < ghosts || i >= ni - ghosts || k < ghosts || k >= nk - ghosts) return;
  } else {
    if (i < ghosts || i >= ni - ghosts || j < ghosts || j >= nj - ghosts) return;
  }

  /* Determine stride and bounds based on direction */
  int stride, npoints_dir, idx_dir;

  if (dir == 0) {
    stride = 1;
    npoints_dir = ni;
    idx_dir = i;
  } else if (dir == 1) {
    stride = ni;
    npoints_dir = nj;
    idx_dir = j;
  } else {
    stride = ni * nj;
    npoints_dir = nk;
    idx_dir = k;
  }

  /* Use tid directly - it already encodes full 3D position */
  int qC = tid;
  int qL = tid - stride;
  int qR = tid + stride;

  for (int v = 0; v < nvars; v++) {
    if (idx_dir == 0) {
      /* Left boundary: forward difference */
      Df[qC*nvars+v] = f[qR*nvars+v] - f[qC*nvars+v];
    } else if (idx_dir == npoints_dir - 1) {
      /* Right boundary: backward difference */
      Df[qC*nvars+v] = f[qC*nvars+v] - f[qL*nvars+v];
    } else {
      /* Interior: biased difference */
      Df[qC*nvars+v] = (bias > 0 ? f[qR*nvars+v] - f[qC*nvars+v] :
                       bias < 0 ? f[qC*nvars+v] - f[qL*nvars+v] :
                       0.5 * (f[qR*nvars+v] - f[qL*nvars+v]));
    }
  }
}

/* Optimized kernel for nvars=5 (Navier-Stokes without species) */
GPU_KERNEL void gpu_first_derivative_first_order_3d_nvars5_kernel(
  double *Df,
  const double *f,
  int ni, int nj, int nk,
  int dir,
  double bias
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  /* Convert linear index to 3D indices */
  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  /* Compute qC, qL, qR as grid point indices (NOT pre-multiplied by nvars) */
  int qC = tid;
  int qL = tid - stride;
  int qR = tid + stride;

  double d0, d1, d2, d3, d4;

  if (idx_dir == 0) {
    d0 = f[qR*5+0] - f[qC*5+0];
    d1 = f[qR*5+1] - f[qC*5+1];
    d2 = f[qR*5+2] - f[qC*5+2];
    d3 = f[qR*5+3] - f[qC*5+3];
    d4 = f[qR*5+4] - f[qC*5+4];
  } else if (idx_dir == npoints_dir - 1) {
    d0 = f[qC*5+0] - f[qL*5+0];
    d1 = f[qC*5+1] - f[qL*5+1];
    d2 = f[qC*5+2] - f[qL*5+2];
    d3 = f[qC*5+3] - f[qL*5+3];
    d4 = f[qC*5+4] - f[qL*5+4];
  } else if (bias > 0) {
    d0 = f[qR*5+0] - f[qC*5+0];
    d1 = f[qR*5+1] - f[qC*5+1];
    d2 = f[qR*5+2] - f[qC*5+2];
    d3 = f[qR*5+3] - f[qC*5+3];
    d4 = f[qR*5+4] - f[qC*5+4];
  } else if (bias < 0) {
    d0 = f[qC*5+0] - f[qL*5+0];
    d1 = f[qC*5+1] - f[qL*5+1];
    d2 = f[qC*5+2] - f[qL*5+2];
    d3 = f[qC*5+3] - f[qL*5+3];
    d4 = f[qC*5+4] - f[qL*5+4];
  } else {
    d0 = 0.5 * (f[qR*5+0] - f[qL*5+0]);
    d1 = 0.5 * (f[qR*5+1] - f[qL*5+1]);
    d2 = 0.5 * (f[qR*5+2] - f[qL*5+2]);
    d3 = 0.5 * (f[qR*5+3] - f[qL*5+3]);
    d4 = 0.5 * (f[qR*5+4] - f[qL*5+4]);
  }

  /* Write output - compute output index separately */
  int qC_out = tid * 5;
  Df[qC_out+0] = d0;
  Df[qC_out+1] = d1;
  Df[qC_out+2] = d2;
  Df[qC_out+3] = d3;
  Df[qC_out+4] = d4;
}

/* Optimized kernel for nvars=12 (Navier-Stokes with 7 species) */
GPU_KERNEL void gpu_first_derivative_first_order_3d_nvars12_kernel(
  double *Df,
  const double *f,
  int ni, int nj, int nk,
  int dir,
  double bias
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  /* Compute qC, qL, qR as in generic kernel (NOT pre-multiplied by nvars) */
  int qC = tid;
  int qL = tid - stride;
  int qR = tid + stride;

  double d[12];

  if (idx_dir == 0) {
    #pragma unroll
    for (int v = 0; v < 12; v++) d[v] = f[qR*12+v] - f[qC*12+v];
  } else if (idx_dir == npoints_dir - 1) {
    #pragma unroll
    for (int v = 0; v < 12; v++) d[v] = f[qC*12+v] - f[qL*12+v];
  } else if (bias > 0) {
    #pragma unroll
    for (int v = 0; v < 12; v++) d[v] = f[qR*12+v] - f[qC*12+v];
  } else if (bias < 0) {
    #pragma unroll
    for (int v = 0; v < 12; v++) d[v] = f[qC*12+v] - f[qL*12+v];
  } else {
    #pragma unroll
    for (int v = 0; v < 12; v++) d[v] = 0.5 * (f[qR*12+v] - f[qL*12+v]);
  }

  /* Write output - compute output index separately */
  int qC_out = tid * 12;
  #pragma unroll
  for (int v = 0; v < 12; v++) Df[qC_out+v] = d[v];
}

/*******************************************************************************
 * 3D BATCHED SECOND-ORDER DERIVATIVE KERNELS
 ******************************************************************************/

/* Kernel: 2nd order central first derivative for entire 3D domain
 * Only processes interior (i,k) positions to match per-line behavior */
GPU_KERNEL void gpu_first_derivative_second_order_3d_kernel(
  double *Df,
  const double *f,
  int nvars,
  int ni, int nj, int nk,
  int ghosts,
  int dir
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  /* Skip ghost regions in non-derivative directions to match per-line behavior */
  if (dir == 0) {
    if (j < ghosts || j >= nj - ghosts || k < ghosts || k >= nk - ghosts) return;
  } else if (dir == 1) {
    if (i < ghosts || i >= ni - ghosts || k < ghosts || k >= nk - ghosts) return;
  } else {
    if (i < ghosts || i >= ni - ghosts || j < ghosts || j >= nj - ghosts) return;
  }

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  int qC = tid;
  int qL = tid - stride;
  int qR = tid + stride;

  for (int v = 0; v < nvars; v++) {
    double deriv;
    if (idx_dir == 0) {
      /* First point: one-sided forward */
      int qRR = tid + 2 * stride;
      deriv = 0.5 * (-3.0*f[qC*nvars+v] + 4.0*f[qR*nvars+v] - f[qRR*nvars+v]);
    } else if (idx_dir == npoints_dir - 1) {
      /* Last point: one-sided backward */
      int qLL = tid - 2 * stride;
      deriv = 0.5 * (3.0*f[qC*nvars+v] - 4.0*f[qL*nvars+v] + f[qLL*nvars+v]);
    } else {
      /* Interior: central difference */
      deriv = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
    }
    Df[qC*nvars+v] = deriv;
  }
}

/* Optimized 2nd order kernel for nvars=5 */
GPU_KERNEL void gpu_first_derivative_second_order_3d_nvars5_kernel(
  double *Df,
  const double *f,
  int ni, int nj, int nk,
  int dir
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  int qC = tid * 5;
  int qL = (tid - stride) * 5;
  int qR = (tid + stride) * 5;

  double d0, d1, d2, d3, d4;

  if (idx_dir == 0) {
    int qRR = (tid + 2 * stride) * 5;
    d0 = 0.5 * (-3.0*f[qC+0] + 4.0*f[qR+0] - f[qRR+0]);
    d1 = 0.5 * (-3.0*f[qC+1] + 4.0*f[qR+1] - f[qRR+1]);
    d2 = 0.5 * (-3.0*f[qC+2] + 4.0*f[qR+2] - f[qRR+2]);
    d3 = 0.5 * (-3.0*f[qC+3] + 4.0*f[qR+3] - f[qRR+3]);
    d4 = 0.5 * (-3.0*f[qC+4] + 4.0*f[qR+4] - f[qRR+4]);
  } else if (idx_dir == npoints_dir - 1) {
    int qLL = (tid - 2 * stride) * 5;
    d0 = 0.5 * (3.0*f[qC+0] - 4.0*f[qL+0] + f[qLL+0]);
    d1 = 0.5 * (3.0*f[qC+1] - 4.0*f[qL+1] + f[qLL+1]);
    d2 = 0.5 * (3.0*f[qC+2] - 4.0*f[qL+2] + f[qLL+2]);
    d3 = 0.5 * (3.0*f[qC+3] - 4.0*f[qL+3] + f[qLL+3]);
    d4 = 0.5 * (3.0*f[qC+4] - 4.0*f[qL+4] + f[qLL+4]);
  } else {
    d0 = 0.5 * (f[qR+0] - f[qL+0]);
    d1 = 0.5 * (f[qR+1] - f[qL+1]);
    d2 = 0.5 * (f[qR+2] - f[qL+2]);
    d3 = 0.5 * (f[qR+3] - f[qL+3]);
    d4 = 0.5 * (f[qR+4] - f[qL+4]);
  }

  Df[qC+0] = d0;
  Df[qC+1] = d1;
  Df[qC+2] = d2;
  Df[qC+3] = d3;
  Df[qC+4] = d4;
}

/* Optimized 2nd order kernel for nvars=12 */
GPU_KERNEL void gpu_first_derivative_second_order_3d_nvars12_kernel(
  double *Df,
  const double *f,
  int ni, int nj, int nk,
  int dir
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  int qC = tid * 12;
  int qL = (tid - stride) * 12;
  int qR = (tid + stride) * 12;

  double d[12];

  if (idx_dir == 0) {
    int qRR = (tid + 2 * stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = 0.5 * (-3.0*f[qC+v] + 4.0*f[qR+v] - f[qRR+v]);
  } else if (idx_dir == npoints_dir - 1) {
    int qLL = (tid - 2 * stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = 0.5 * (3.0*f[qC+v] - 4.0*f[qL+v] + f[qLL+v]);
  } else {
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = 0.5 * (f[qR+v] - f[qL+v]);
  }

  #pragma unroll
  for (int v = 0; v < 12; v++) Df[qC+v] = d[v];
}

/*******************************************************************************
 * 3D BATCHED FOURTH-ORDER DERIVATIVE KERNELS
 ******************************************************************************/

/* Kernel: 4th order central first derivative for entire 3D domain
 * Only processes interior (i,k) positions to match per-line behavior */
GPU_KERNEL void gpu_first_derivative_fourth_order_3d_kernel(
  double *Df,
  const double *f,
  int nvars,
  int ni, int nj, int nk,
  int ghosts,
  int dir
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  /* Skip ghost regions in non-derivative directions to match per-line behavior */
  if (dir == 0) {
    if (j < ghosts || j >= nj - ghosts || k < ghosts || k >= nk - ghosts) return;
  } else if (dir == 1) {
    if (i < ghosts || i >= ni - ghosts || k < ghosts || k >= nk - ghosts) return;
  } else {
    if (i < ghosts || i >= ni - ghosts || j < ghosts || j >= nj - ghosts) return;
  }

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  const double one_twelve = 1.0/12.0;
  int qC = tid;

  for (int v = 0; v < nvars; v++) {
    double deriv;
    if (idx_dir == 0) {
      /* First point: 4th order forward biased */
      int qp1 = tid + stride;
      int qp2 = tid + 2*stride;
      int qp3 = tid + 3*stride;
      int qp4 = tid + 4*stride;
      deriv = one_twelve * (-25.0*f[qC*nvars+v] + 48.0*f[qp1*nvars+v] - 36.0*f[qp2*nvars+v] + 16.0*f[qp3*nvars+v] - 3.0*f[qp4*nvars+v]);
    } else if (idx_dir == 1) {
      /* Second point: 4th order forward biased */
      int qm1 = tid - stride;
      int qp1 = tid + stride;
      int qp2 = tid + 2*stride;
      int qp3 = tid + 3*stride;
      deriv = one_twelve * (-3.0*f[qm1*nvars+v] - 10.0*f[qC*nvars+v] + 18.0*f[qp1*nvars+v] - 6.0*f[qp2*nvars+v] + f[qp3*nvars+v]);
    } else if (idx_dir >= 2 && idx_dir < npoints_dir - 2) {
      /* Interior: 4th order central */
      int qL = tid - stride;
      int qR = tid + stride;
      int qLL = tid - 2*stride;
      int qRR = tid + 2*stride;
      deriv = one_twelve * (f[qLL*nvars+v] - 8.0*f[qL*nvars+v] + 8.0*f[qR*nvars+v] - f[qRR*nvars+v]);
    } else if (idx_dir == npoints_dir - 2) {
      /* Second-to-last: 4th order backward biased */
      int qm3 = tid - 3*stride;
      int qm2 = tid - 2*stride;
      int qm1 = tid - stride;
      int qp1 = tid + stride;
      deriv = one_twelve * (-f[qm3*nvars+v] + 6.0*f[qm2*nvars+v] - 18.0*f[qm1*nvars+v] + 10.0*f[qC*nvars+v] + 3.0*f[qp1*nvars+v]);
    } else {
      /* Last point: 4th order backward biased */
      int qm4 = tid - 4*stride;
      int qm3 = tid - 3*stride;
      int qm2 = tid - 2*stride;
      int qm1 = tid - stride;
      deriv = one_twelve * (3.0*f[qm4*nvars+v] - 16.0*f[qm3*nvars+v] + 36.0*f[qm2*nvars+v] - 48.0*f[qm1*nvars+v] + 25.0*f[qC*nvars+v]);
    }
    Df[qC*nvars+v] = deriv;
  }
}

/* Optimized 4th order kernel for nvars=5 */
GPU_KERNEL void gpu_first_derivative_fourth_order_3d_nvars5_kernel(
  double *Df,
  const double *f,
  int ni, int nj, int nk,
  int dir
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  const double one_twelve = 1.0/12.0;
  int qC = tid * 5;
  double d0, d1, d2, d3, d4;

  if (idx_dir == 0) {
    int qp1 = (tid + stride) * 5;
    int qp2 = (tid + 2*stride) * 5;
    int qp3 = (tid + 3*stride) * 5;
    int qp4 = (tid + 4*stride) * 5;
    d0 = one_twelve * (-25.0*f[qC+0] + 48.0*f[qp1+0] - 36.0*f[qp2+0] + 16.0*f[qp3+0] - 3.0*f[qp4+0]);
    d1 = one_twelve * (-25.0*f[qC+1] + 48.0*f[qp1+1] - 36.0*f[qp2+1] + 16.0*f[qp3+1] - 3.0*f[qp4+1]);
    d2 = one_twelve * (-25.0*f[qC+2] + 48.0*f[qp1+2] - 36.0*f[qp2+2] + 16.0*f[qp3+2] - 3.0*f[qp4+2]);
    d3 = one_twelve * (-25.0*f[qC+3] + 48.0*f[qp1+3] - 36.0*f[qp2+3] + 16.0*f[qp3+3] - 3.0*f[qp4+3]);
    d4 = one_twelve * (-25.0*f[qC+4] + 48.0*f[qp1+4] - 36.0*f[qp2+4] + 16.0*f[qp3+4] - 3.0*f[qp4+4]);
  } else if (idx_dir == 1) {
    int qm1 = (tid - stride) * 5;
    int qp1 = (tid + stride) * 5;
    int qp2 = (tid + 2*stride) * 5;
    int qp3 = (tid + 3*stride) * 5;
    d0 = one_twelve * (-3.0*f[qm1+0] - 10.0*f[qC+0] + 18.0*f[qp1+0] - 6.0*f[qp2+0] + f[qp3+0]);
    d1 = one_twelve * (-3.0*f[qm1+1] - 10.0*f[qC+1] + 18.0*f[qp1+1] - 6.0*f[qp2+1] + f[qp3+1]);
    d2 = one_twelve * (-3.0*f[qm1+2] - 10.0*f[qC+2] + 18.0*f[qp1+2] - 6.0*f[qp2+2] + f[qp3+2]);
    d3 = one_twelve * (-3.0*f[qm1+3] - 10.0*f[qC+3] + 18.0*f[qp1+3] - 6.0*f[qp2+3] + f[qp3+3]);
    d4 = one_twelve * (-3.0*f[qm1+4] - 10.0*f[qC+4] + 18.0*f[qp1+4] - 6.0*f[qp2+4] + f[qp3+4]);
  } else if (idx_dir >= 2 && idx_dir < npoints_dir - 2) {
    int qL = (tid - stride) * 5;
    int qR = (tid + stride) * 5;
    int qLL = (tid - 2*stride) * 5;
    int qRR = (tid + 2*stride) * 5;
    d0 = one_twelve * (f[qLL+0] - 8.0*f[qL+0] + 8.0*f[qR+0] - f[qRR+0]);
    d1 = one_twelve * (f[qLL+1] - 8.0*f[qL+1] + 8.0*f[qR+1] - f[qRR+1]);
    d2 = one_twelve * (f[qLL+2] - 8.0*f[qL+2] + 8.0*f[qR+2] - f[qRR+2]);
    d3 = one_twelve * (f[qLL+3] - 8.0*f[qL+3] + 8.0*f[qR+3] - f[qRR+3]);
    d4 = one_twelve * (f[qLL+4] - 8.0*f[qL+4] + 8.0*f[qR+4] - f[qRR+4]);
  } else if (idx_dir == npoints_dir - 2) {
    int qm3 = (tid - 3*stride) * 5;
    int qm2 = (tid - 2*stride) * 5;
    int qm1 = (tid - stride) * 5;
    int qp1 = (tid + stride) * 5;
    d0 = one_twelve * (-f[qm3+0] + 6.0*f[qm2+0] - 18.0*f[qm1+0] + 10.0*f[qC+0] + 3.0*f[qp1+0]);
    d1 = one_twelve * (-f[qm3+1] + 6.0*f[qm2+1] - 18.0*f[qm1+1] + 10.0*f[qC+1] + 3.0*f[qp1+1]);
    d2 = one_twelve * (-f[qm3+2] + 6.0*f[qm2+2] - 18.0*f[qm1+2] + 10.0*f[qC+2] + 3.0*f[qp1+2]);
    d3 = one_twelve * (-f[qm3+3] + 6.0*f[qm2+3] - 18.0*f[qm1+3] + 10.0*f[qC+3] + 3.0*f[qp1+3]);
    d4 = one_twelve * (-f[qm3+4] + 6.0*f[qm2+4] - 18.0*f[qm1+4] + 10.0*f[qC+4] + 3.0*f[qp1+4]);
  } else {
    int qm4 = (tid - 4*stride) * 5;
    int qm3 = (tid - 3*stride) * 5;
    int qm2 = (tid - 2*stride) * 5;
    int qm1 = (tid - stride) * 5;
    d0 = one_twelve * (3.0*f[qm4+0] - 16.0*f[qm3+0] + 36.0*f[qm2+0] - 48.0*f[qm1+0] + 25.0*f[qC+0]);
    d1 = one_twelve * (3.0*f[qm4+1] - 16.0*f[qm3+1] + 36.0*f[qm2+1] - 48.0*f[qm1+1] + 25.0*f[qC+1]);
    d2 = one_twelve * (3.0*f[qm4+2] - 16.0*f[qm3+2] + 36.0*f[qm2+2] - 48.0*f[qm1+2] + 25.0*f[qC+2]);
    d3 = one_twelve * (3.0*f[qm4+3] - 16.0*f[qm3+3] + 36.0*f[qm2+3] - 48.0*f[qm1+3] + 25.0*f[qC+3]);
    d4 = one_twelve * (3.0*f[qm4+4] - 16.0*f[qm3+4] + 36.0*f[qm2+4] - 48.0*f[qm1+4] + 25.0*f[qC+4]);
  }

  Df[qC+0] = d0;
  Df[qC+1] = d1;
  Df[qC+2] = d2;
  Df[qC+3] = d3;
  Df[qC+4] = d4;
}

/* Optimized 4th order kernel for nvars=12 */
GPU_KERNEL void gpu_first_derivative_fourth_order_3d_nvars12_kernel(
  double *Df,
  const double *f,
  int ni, int nj, int nk,
  int dir
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = ni * nj * nk;

  if (tid >= total_points) return;

  int i = tid % ni;
  int j = (tid / ni) % nj;
  int k = tid / (ni * nj);

  int stride, npoints_dir, idx_dir;
  if (dir == 0) { stride = 1; npoints_dir = ni; idx_dir = i; }
  else if (dir == 1) { stride = ni; npoints_dir = nj; idx_dir = j; }
  else { stride = ni * nj; npoints_dir = nk; idx_dir = k; }

  const double one_twelve = 1.0/12.0;
  int qC = tid * 12;
  double d[12];

  if (idx_dir == 0) {
    int qp1 = (tid + stride) * 12;
    int qp2 = (tid + 2*stride) * 12;
    int qp3 = (tid + 3*stride) * 12;
    int qp4 = (tid + 4*stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = one_twelve * (-25.0*f[qC+v] + 48.0*f[qp1+v] - 36.0*f[qp2+v] + 16.0*f[qp3+v] - 3.0*f[qp4+v]);
  } else if (idx_dir == 1) {
    int qm1 = (tid - stride) * 12;
    int qp1 = (tid + stride) * 12;
    int qp2 = (tid + 2*stride) * 12;
    int qp3 = (tid + 3*stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = one_twelve * (-3.0*f[qm1+v] - 10.0*f[qC+v] + 18.0*f[qp1+v] - 6.0*f[qp2+v] + f[qp3+v]);
  } else if (idx_dir >= 2 && idx_dir < npoints_dir - 2) {
    int qL = (tid - stride) * 12;
    int qR = (tid + stride) * 12;
    int qLL = (tid - 2*stride) * 12;
    int qRR = (tid + 2*stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = one_twelve * (f[qLL+v] - 8.0*f[qL+v] + 8.0*f[qR+v] - f[qRR+v]);
  } else if (idx_dir == npoints_dir - 2) {
    int qm3 = (tid - 3*stride) * 12;
    int qm2 = (tid - 2*stride) * 12;
    int qm1 = (tid - stride) * 12;
    int qp1 = (tid + stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = one_twelve * (-f[qm3+v] + 6.0*f[qm2+v] - 18.0*f[qm1+v] + 10.0*f[qC+v] + 3.0*f[qp1+v]);
  } else {
    int qm4 = (tid - 4*stride) * 12;
    int qm3 = (tid - 3*stride) * 12;
    int qm2 = (tid - 2*stride) * 12;
    int qm1 = (tid - stride) * 12;
    #pragma unroll
    for (int v = 0; v < 12; v++)
      d[v] = one_twelve * (3.0*f[qm4+v] - 16.0*f[qm3+v] + 36.0*f[qm2+v] - 48.0*f[qm1+v] + 25.0*f[qC+v]);
  }

  #pragma unroll
  for (int v = 0; v < 12; v++) Df[qC+v] = d[v];
}

