/*! @file gpu_derivative.h
    @brief GPU derivative kernel declarations
*/

#ifndef _GPU_DERIVATIVE_H_
#define _GPU_DERIVATIVE_H_

#include <gpu.h>

/* Kernel annotations must only be visible to CUDA/HIP compilers. */
#if defined(GPU_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP) && (defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__))
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel declarations */
GPU_KERNEL void gpu_first_derivative_second_order_kernel(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride
);

GPU_KERNEL void gpu_first_derivative_fourth_order_kernel(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride
);

GPU_KERNEL void gpu_first_derivative_first_order_kernel(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, double bias
);

/* Launch wrappers */
#ifdef __cplusplus
extern "C" {
#endif

void gpu_launch_first_derivative_second_order(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, int blockSize
);

void gpu_launch_first_derivative_fourth_order(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, int blockSize
);

void gpu_launch_first_derivative_first_order(
  double *Df, const double *f, int nvars, int npoints, int ghosts, int stride, double bias, int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_DERIVATIVE_H_ */

