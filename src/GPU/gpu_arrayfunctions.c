/*! @file gpu_arrayfunctions.c
    @brief GPU-enabled array function implementations
*/

#include <gpu.h>
#include <gpu_arrayfunctions.h>
#include <gpu_launch.h>  /* Always needed for function declarations */
#include <gpu_runtime.h>

#define DEFAULT_BLOCK_SIZE 256

void GPUArraySetValue(double *x, double value, int n)
{
  /* Clear any previous errors before launching kernel */
  GPU_GET_LAST_ERROR();
  
  gpu_launch_array_set_value(x, value, n, DEFAULT_BLOCK_SIZE);
  
  /* Check for errors after kernel launch */
  int err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPUArraySetValue failed (n=%d, error=%d)\n", n, err);
    exit(1);
  }
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayCopy(double *dst, const double *src, int n)
{
  /* Clear any previous errors before launching kernel */
  GPU_GET_LAST_ERROR();
  
  gpu_launch_array_copy(dst, src, n, DEFAULT_BLOCK_SIZE);
  
  /* Check for errors after kernel launch */
  int err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPUArrayCopy failed (n=%d, error=%d, dst=%p, src=%p)\n", n, err, dst, src);
    exit(1);
  }
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayScale(double *x, double a, int n)
{
  gpu_launch_array_scale(x, a, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayAXPY(const double *x, double a, double *y, int n)
{
  gpu_launch_array_axpy(x, a, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayAYPX(const double *x, double a, double *y, int n)
{
  gpu_launch_array_aypx(x, a, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayAXBY(double *z, double a, const double *x, double b, const double *y, int n)
{
  gpu_launch_array_axby(z, a, x, b, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayScaleCopy(const double *x, double a, double *y, int n)
{
  gpu_launch_array_scale_copy(x, a, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayAdd(double *z, const double *x, const double *y, int n)
{
  gpu_launch_array_add(z, x, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArraySubtract(double *z, const double *x, const double *y, int n)
{
  gpu_launch_array_subtract(z, x, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

void GPUArrayMultiply(double *z, const double *x, const double *y, int n)
{
  gpu_launch_array_multiply(z, x, y, n, DEFAULT_BLOCK_SIZE);
  if (GPUShouldSyncEveryOp()) GPUSync();
}

int GPUIsDevicePtr(const void *ptr)
{
#ifdef GPU_NONE
  return 0;
#else
  return 1; /* In GPU mode, assume all pointers are device pointers */
#endif
}

