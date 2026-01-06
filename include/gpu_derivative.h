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

/* 3D batched derivative kernels - process entire domain in single launch */

/* First-order 3D kernels - ghosts parameter to skip ghost regions in non-derivative directions */
GPU_KERNEL void gpu_first_derivative_first_order_3d_kernel(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, double bias
);

GPU_KERNEL void gpu_first_derivative_first_order_3d_nvars5_kernel(
  double *Df, const double *f, int ni, int nj, int nk, int ghosts, int dir, double bias
);

GPU_KERNEL void gpu_first_derivative_first_order_3d_nvars12_kernel(
  double *Df, const double *f, int ni, int nj, int nk, int ghosts, int dir, double bias
);

/* Second-order 3D kernels */
GPU_KERNEL void gpu_first_derivative_second_order_3d_kernel(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir
);

GPU_KERNEL void gpu_first_derivative_second_order_3d_nvars5_kernel(
  double *Df, const double *f, int ni, int nj, int nk, int ghosts, int dir
);

GPU_KERNEL void gpu_first_derivative_second_order_3d_nvars12_kernel(
  double *Df, const double *f, int ni, int nj, int nk, int ghosts, int dir
);

/* Fourth-order 3D kernels */
GPU_KERNEL void gpu_first_derivative_fourth_order_3d_kernel(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir
);

GPU_KERNEL void gpu_first_derivative_fourth_order_3d_nvars5_kernel(
  double *Df, const double *f, int ni, int nj, int nk, int ghosts, int dir
);

GPU_KERNEL void gpu_first_derivative_fourth_order_3d_nvars12_kernel(
  double *Df, const double *f, int ni, int nj, int nk, int ghosts, int dir
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

/* 3D batched launch wrappers - ghosts parameter to skip ghost regions in non-derivative directions */
void gpu_launch_first_derivative_first_order_3d(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, double bias, int blockSize
);

void gpu_launch_first_derivative_second_order_3d(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, int blockSize
);

void gpu_launch_first_derivative_fourth_order_3d(
  double *Df, const double *f, int nvars, int ni, int nj, int nk, int ghosts, int dir, int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_DERIVATIVE_H_ */

