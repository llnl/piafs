/*! @file gpu_kernels_optimized.cu
    @brief Optimized GPU kernels with memory coalescing and kernel fusion
*/

#include <gpu.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel fusion: combined operations to reduce memory bandwidth */

/* Fused AXPY chain: y = a1*x1 + a2*x2 + y */
GPU_KERNEL void gpu_array_axpy_chain2(
  const double * __restrict__ x1, double a1,
  const double * __restrict__ x2, double a2,
  double * __restrict__ y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a1 * x1[idx] + a2 * x2[idx];
  }
}

/* Fused AXPY chain: y = a1*x1 + a2*x2 + a3*x3 + y */
GPU_KERNEL void gpu_array_axpy_chain3(
  const double * __restrict__ x1, double a1,
  const double * __restrict__ x2, double a2,
  const double * __restrict__ x3, double a3,
  double * __restrict__ y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a1 * x1[idx] + a2 * x2[idx] + a3 * x3[idx];
  }
}

/* Fused AXPY chain: y = a1*x1 + a2*x2 + a3*x3 + a4*x4 + y */
GPU_KERNEL void gpu_array_axpy_chain4(
  const double * __restrict__ x1, double a1,
  const double * __restrict__ x2, double a2,
  const double * __restrict__ x3, double a3,
  const double * __restrict__ x4, double a4,
  double * __restrict__ y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a1 * x1[idx] + a2 * x2[idx] + a3 * x3[idx] + a4 * x4[idx];
  }
}

/* General fused AXPY chain: y = sum(a[i]*x[i]) + y for arbitrary number of arrays */
GPU_KERNEL void gpu_array_axpy_chain_general(
  const double ** __restrict__ x_arrays,  /* Array of pointers to source arrays */
  const double * __restrict__ coeffs,     /* Array of coefficients */
  double * __restrict__ y,                /* Destination array */
  int n_arrays,                           /* Number of source arrays */
  int n)                                  /* Array size */
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double sum = 0.0;
    for (int i = 0; i < n_arrays; i++) {
      sum += coeffs[i] * x_arrays[i][idx];
    }
    y[idx] += sum;
  }
}

/* Fused scale and add: z = a*x + b*y (more efficient than separate ops) */
GPU_KERNEL void gpu_array_axpby_fused(
  double * __restrict__ z,
  double a, const double * __restrict__ x,
  double b, const double * __restrict__ y,
  int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = a * x[idx] + b * y[idx];
  }
}

/* Optimized array copy with restrict for better vectorization */
GPU_KERNEL void gpu_array_copy_optimized(
  double * __restrict__ dst,
  const double * __restrict__ src,
  int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

/* Optimized AXPY with restrict */
GPU_KERNEL void gpu_array_axpy_optimized(
  const double * __restrict__ x,
  double a,
  double * __restrict__ y,
  int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a * x[idx];
  }
}

/* Launch wrappers */
extern "C" {

void gpu_launch_array_axpy_chain2(
  const double *x1, double a1,
  const double *x2, double a2,
  double *y, int n, int blockSize)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = 512;  /* Optimized default */
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axpy_chain2, gridSize, blockSize)(x1, a1, x2, a2, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  for (int i = 0; i < n; i++) {
    y[i] += a1 * x1[i] + a2 * x2[i];
  }
#endif
}

void gpu_launch_array_axpy_chain3(
  const double *x1, double a1,
  const double *x2, double a2,
  const double *x3, double a3,
  double *y, int n, int blockSize)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = 512;  /* Optimized default */
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axpy_chain3, gridSize, blockSize)(x1, a1, x2, a2, x3, a3, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  for (int i = 0; i < n; i++) {
    y[i] += a1 * x1[i] + a2 * x2[i] + a3 * x3[i];
  }
#endif
}

void gpu_launch_array_axpy_chain4(
  const double *x1, double a1,
  const double *x2, double a2,
  const double *x3, double a3,
  const double *x4, double a4,
  double *y, int n, int blockSize)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = 512;  /* Optimized default */
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axpy_chain4, gridSize, blockSize)(x1, a1, x2, a2, x3, a3, x4, a4, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  for (int i = 0; i < n; i++) {
    y[i] += a1 * x1[i] + a2 * x2[i] + a3 * x3[i] + a4 * x4[i];
  }
#endif
}

void gpu_launch_array_axpy_chain_general(
  const double **x_arrays,
  const double *coeffs,
  double *y,
  int n_arrays,
  int n,
  int blockSize)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = 512;  /* Optimized default */
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axpy_chain_general, gridSize, blockSize)(x_arrays, coeffs, y, n_arrays, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n_arrays; j++) {
      sum += coeffs[j] * x_arrays[j][i];
    }
    y[i] += sum;
  }
#endif
}

void gpu_launch_array_axpby_fused(
  double *z, double a, const double *x,
  double b, const double *y, int n, int blockSize)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = 512;  /* Optimized default */
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axpby_fused, gridSize, blockSize)(z, a, x, b, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  for (int i = 0; i < n; i++) {
    z[i] = a * x[i] + b * y[i];
  }
#endif
}

} /* extern "C" */
