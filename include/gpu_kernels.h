/*! @file gpu_kernels.h
    @brief GPU kernel declarations
*/

#ifndef _GPU_KERNELS_H_
#define _GPU_KERNELS_H_

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
GPU_KERNEL void gpu_array_copy(double *dst, const double *src, int n);
GPU_KERNEL void gpu_array_set_value(double *x, double value, int n);
GPU_KERNEL void gpu_array_scale(double *x, double a, int n);
GPU_KERNEL void gpu_array_axpy(const double *x, double a, double *y, int n);
GPU_KERNEL void gpu_array_aypx(const double *x, double a, double *y, int n);
GPU_KERNEL void gpu_array_axby(double *z, double a, const double *x, double b, const double *y, int n);
GPU_KERNEL void gpu_array_scale_copy(const double *x, double a, double *y, int n);
GPU_KERNEL void gpu_array_add(double *z, const double *x, const double *y, int n);
GPU_KERNEL void gpu_array_subtract(double *z, const double *x, const double *y, int n);
GPU_KERNEL void gpu_array_multiply(double *z, const double *x, const double *y, int n);
GPU_KERNEL void gpu_array_max_block(const double *x, double *partial_max, int n);
GPU_KERNEL void gpu_array_max_final(const double *partial_max, double *result, int n);
GPU_KERNEL void gpu_hyperbolic_flux_derivative(
  double *hyp, const double *fluxI, const double *dxinv,
  int nvars, int ndims, int *dim, int ghosts, int dir, int offset
);

/* Kernel launch wrappers */
void gpu_launch_array_copy(double *dst, const double *src, int n, int blockSize);
void gpu_launch_array_set_value(double *x, double value, int n, int blockSize);
void gpu_launch_array_scale(double *x, double a, int n, int blockSize);
void gpu_launch_array_axpy(const double *x, double a, double *y, int n, int blockSize);
void gpu_launch_array_aypx(const double *x, double a, double *y, int n, int blockSize);
void gpu_launch_array_axby(double *z, double a, const double *x, double b, const double *y, int n, int blockSize);
void gpu_launch_array_scale_copy(const double *x, double a, double *y, int n, int blockSize);
void gpu_launch_array_add(double *z, const double *x, const double *y, int n, int blockSize);
void gpu_launch_array_subtract(double *z, const double *x, const double *y, int n, int blockSize);
void gpu_launch_array_multiply(double *z, const double *x, const double *y, int n, int blockSize);
double gpu_launch_array_max(const double *x, int n, int blockSize);

/* Fused kernel operations for performance */
void gpu_launch_array_axpy_chain2(
  const double *x1, double a1,
  const double *x2, double a2,
  double *y, int n, int blockSize);

void gpu_launch_array_axpy_chain3(
  const double *x1, double a1,
  const double *x2, double a2,
  const double *x3, double a3,
  double *y, int n, int blockSize);

void gpu_launch_array_axpy_chain4(
  const double *x1, double a1,
  const double *x2, double a2,
  const double *x3, double a3,
  const double *x4, double a4,
  double *y, int n, int blockSize);

void gpu_launch_array_axpy_chain_general(
  const double **x_arrays,
  const double *coeffs,
  double *y,
  int n_arrays,
  int n,
  int blockSize);

#endif /* _GPU_KERNELS_H_ */

