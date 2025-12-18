/*! @file gpu_launch.cu
    @brief GPU kernel launch wrappers
*/

#include <gpu.h>
#include <gpu_launch.h>
#ifndef GPU_NONE
#include <gpu_kernels.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

extern "C" {

void gpu_launch_array_copy(double *dst, const double *src, int n, int blockSize)
{
#ifdef GPU_NONE
  /* CPU fallback */
  for (int i = 0; i < n; i++) dst[i] = src[i];
#else
  /* Clear any previous errors */
  GPU_GET_LAST_ERROR();
  
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  
  /* Safety checks */
  if (!dst || !src) {
    fprintf(stderr, "Error: gpu_launch_array_copy: NULL pointer (dst=%p, src=%p, n=%d)\n", dst, src, n);
    exit(1);
  }
  if (n <= 0) {
    fprintf(stderr, "Error: gpu_launch_array_copy: invalid size n=%d\n", n);
    exit(1);
  }
  
  GPU_KERNEL_LAUNCH(gpu_array_copy, gridSize, blockSize)(dst, src, n);
  
  /* Check for errors immediately after launch */
  int err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: gpu_launch_array_copy failed: dst=%p, src=%p, n=%d, gridSize=%d, blockSize=%d, error=%d\n",
            dst, src, n, gridSize, blockSize, err);
    exit(1);
  }
#endif
}

void gpu_launch_array_set_value(double *x, double value, int n, int blockSize)
{
#ifdef GPU_NONE
  for (int i = 0; i < n; i++) x[i] = value;
#else
  /* Clear any previous errors */
  GPU_GET_LAST_ERROR();
  
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (n + blockSize - 1) / blockSize;
  
  /* Safety checks */
  if (!x) {
    fprintf(stderr, "Error: gpu_launch_array_set_value: NULL pointer (x=%p, n=%d)\n", x, n);
    exit(1);
  }
  if (n <= 0) {
    fprintf(stderr, "Error: gpu_launch_array_set_value: invalid size n=%d\n", n);
    exit(1);
  }
  
  GPU_KERNEL_LAUNCH(gpu_array_set_value, gridSize, blockSize)(x, value, n);
  
  /* Check for errors immediately after launch */
  int err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: gpu_launch_array_set_value failed: x=%p, n=%d, gridSize=%d, blockSize=%d, error=%d\n",
            x, n, gridSize, blockSize, err);
    exit(1);
  }
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

/* GPU max reduction - returns the maximum value in array x of length n
   Uses a two-pass reduction: first pass computes partial maxes per block,
   second pass reduces partial maxes to final result.
*/
double gpu_launch_array_max(const double *x, int n, int blockSize)
{
#ifdef GPU_NONE
  /* CPU fallback */
  double max_val = -1e308;
  for (int i = 0; i < n; i++) {
    if (x[i] > max_val) max_val = x[i];
  }
  return max_val;
#else
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  if (n <= 0) return 0.0;

  int gridSize = (n + blockSize - 1) / blockSize;

  /* Allocate device memory for partial results */
  double *d_partial = NULL;
  double *d_result = NULL;
  GPU_MALLOC((void**)&d_partial, gridSize * sizeof(double));
  GPU_MALLOC((void**)&d_result, sizeof(double));

  /* First pass: each block computes its partial max */
  size_t sharedMemSize = blockSize * sizeof(double);
  gpu_array_max_block<<<gridSize, blockSize, sharedMemSize>>>(x, d_partial, n);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());

  /* Second pass: reduce partial results (single block is usually enough) */
  int finalBlockSize = (gridSize < blockSize) ? gridSize : blockSize;
  /* Round up to next power of 2 for efficient reduction */
  int powerOf2 = 1;
  while (powerOf2 < finalBlockSize) powerOf2 <<= 1;
  finalBlockSize = powerOf2;
  if (finalBlockSize > 1024) finalBlockSize = 1024; /* Max block size */

  sharedMemSize = finalBlockSize * sizeof(double);
  gpu_array_max_final<<<1, finalBlockSize, sharedMemSize>>>(d_partial, d_result, gridSize);
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());

  /* Copy result back to host */
  double result;
  GPU_MEMCPY(&result, d_result, sizeof(double), GPU_MEMCPY_D2H);

  /* Free temporary arrays */
  GPU_FREE(d_partial);
  GPU_FREE(d_result);

  return result;
#endif
}

} /* extern "C" */

