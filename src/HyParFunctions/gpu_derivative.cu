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
  
  int i = idx - ghosts;  /* convert to interior index */
  
  if (i >= 0 && i < npoints - 2*ghosts) {
    int qC = idx * stride;  /* current point */
    int qL = (idx - 1) * stride;  /* left neighbor */
    int qR = (idx + 1) * stride;  /* right neighbor */
    
    for (int v = 0; v < nvars; v++) {
      if (i == 0) {
        /* Left boundary: one-sided */
        int qRR = (idx + 2) * stride;
        Df[qC*nvars+v] = 0.5 * (-3*f[qC*nvars+v] + 4*f[qR*nvars+v] - f[qRR*nvars+v]);
      } else if (i == npoints - 2*ghosts - 1) {
        /* Right boundary: one-sided */
        int qLL = (idx - 2) * stride;
        Df[qC*nvars+v] = 0.5 * (3*f[qC*nvars+v] - 4*f[qL*nvars+v] + f[qLL*nvars+v]);
      } else {
        /* Interior: central difference */
        Df[qC*nvars+v] = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
      }
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
  
  int i = idx - ghosts;  /* convert to interior index */
  
  if (i >= 0 && i < npoints - 2*ghosts) {
    int qC = idx * stride;
    int qL = (idx - 1) * stride;
    int qR = (idx + 1) * stride;
    int qLL = (idx - 2) * stride;
    int qRR = (idx + 2) * stride;
    
    static const double c0 = -1.0/12.0;
    static const double c1 = 2.0/3.0;
    static const double c2 = -2.0/3.0;
    static const double c3 = 1.0/12.0;
    
    for (int v = 0; v < nvars; v++) {
      if (i == 0 || i == npoints - 2*ghosts - 1) {
        /* Boundary: fall back to 2nd order */
        Df[qC*nvars+v] = 0.5 * (f[qR*nvars+v] - f[qL*nvars+v]);
      } else {
        /* Interior: 4th order central */
        Df[qC*nvars+v] = c0*f[qLL*nvars+v] + c1*f[qL*nvars+v] + c2*f[qR*nvars+v] + c3*f[qRR*nvars+v];
      }
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
  int i = idx - ghosts;
  
  if (i >= 0 && i < npoints - 2*ghosts) {
    int qC = idx;
    int qL = idx - stride;
    int qR = idx + stride;
    
    for (int v = 0; v < nvars; v++) {
      if (i == 0) {
        /* Left boundary */
        Df[qC*nvars+v] = f[qR*nvars+v] - f[qC*nvars+v];
      } else if (i == npoints - 2*ghosts - 1) {
        /* Right boundary */
        Df[qC*nvars+v] = f[qC*nvars+v] - f[qL*nvars+v];
      } else {
        /* Interior: biased difference */
        Df[qC*nvars+v] = (bias > 0 ? f[qR*nvars+v] - f[qC*nvars+v] :
                         bias < 0 ? f[qC*nvars+v] - f[qL*nvars+v] :
                         0.5 * (f[qR*nvars+v] - f[qL*nvars+v]));
      }
    }
  }
}

