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

