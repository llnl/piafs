/*! @file gpu_launch.c
    @brief GPU kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_launch.h>
#ifndef GPU_NONE
#include <gpu_kernels.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_array_copy(double *dst, const double *src, int n, int blockSize)
{
#ifdef GPU_NONE
  /* CPU fallback */
  for (int i = 0; i < n; i++) dst[i] = src[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_copy, gridSize, blockSize)(dst, src, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_set_value(double *x, double value, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) x[i] = value;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_set_value, gridSize, blockSize)(x, value, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_scale(double *x, double a, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) x[i] *= a;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_scale, gridSize, blockSize)(x, a, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_axpy(const double *x, double a, double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) y[i] += a * x[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axpy, gridSize, blockSize)(x, a, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_aypx(const double *x, double a, double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) y[i] = a * y[i] + x[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_aypx, gridSize, blockSize)(x, a, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_axby(double *z, double a, const double *x, double b, const double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) z[i] = a * x[i] + b * y[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_axby, gridSize, blockSize)(z, a, x, b, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_scale_copy(const double *x, double a, double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) y[i] = a * x[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_scale_copy, gridSize, blockSize)(x, a, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_add(double *z, const double *x, const double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) z[i] = x[i] + y[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_add, gridSize, blockSize)(z, x, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_subtract(double *z, const double *x, const double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) z[i] = x[i] - y[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_subtract, gridSize, blockSize)(z, x, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

void gpu_launch_array_multiply(double *z, const double *x, const double *y, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) z[i] = x[i] * y[i];
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_array_multiply, gridSize, blockSize)(z, x, y, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

