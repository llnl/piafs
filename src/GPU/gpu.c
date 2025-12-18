/*! @file gpu.c
    @brief GPU abstraction layer implementation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gpu.h>

/* GPU memory management */
int GPUAllocate(void **ptr, size_t size)
{
#ifdef GPU_NONE
  *ptr = malloc(size);
  return (*ptr == NULL) ? 1 : 0;
#else
  /* Clear any previous errors before allocation */
  int prev_err = GPU_GET_LAST_ERROR();
  if (prev_err != GPU_SUCCESS) {
    fprintf(stderr, "Warning: Previous GPU error detected before allocation: %d\n", prev_err);
#ifdef GPU_CUDA
    fprintf(stderr, "  CUDA error: %s\n", cudaGetErrorString((cudaError_t)prev_err));
#elif defined(GPU_HIP)
    fprintf(stderr, "  HIP error: %s\n", hipGetErrorString((hipError_t)prev_err));
#endif
  }
  
  int err = GPU_MALLOC(ptr, size);
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPU memory allocation failed (size: %zu bytes, error: %d)\n", size, err);
#ifdef GPU_CUDA
    fprintf(stderr, "  CUDA error: %s\n", cudaGetErrorString((cudaError_t)err));
#elif defined(GPU_HIP)
    fprintf(stderr, "  HIP error: %s\n", hipGetErrorString((hipError_t)err));
#endif
    return 1;
  }
  return 0;
#endif
}

int GPUFree(void *ptr)
{
  if (!ptr) return 0;
#ifdef GPU_NONE
  free(ptr);
  return 0;
#else
  /* Clear any previous errors before freeing - sticky errors can interfere with free operations */
  (void)GPU_GET_LAST_ERROR(); /* This clears the error state in CUDA/HIP */
  
  int err = GPU_FREE(ptr);
  if (err != GPU_SUCCESS) {
    /* Note: GPU memory free errors during cleanup are often benign and can be safely ignored */
    return 1;
  }
  return 0;
#endif
}

int GPUCopyToDevice(void *dst, const void *src, size_t size)
{
#ifdef GPU_NONE
  memcpy(dst, src, size);
  return 0;
#else
  int err = GPU_MEMCPY(dst, src, size, GPU_MEMCPY_H2D);
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPU copy host-to-device failed (size: %zu bytes)\n", size);
    return 1;
  }
  return 0;
#endif
}

int GPUCopyToHost(void *dst, const void *src, size_t size)
{
#ifdef GPU_NONE
  memcpy(dst, src, size);
  return 0;
#else
  /* Clear any previous errors */
  GPU_GET_LAST_ERROR();
  
  int err = GPU_MEMCPY(dst, src, size, GPU_MEMCPY_D2H);
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPU copy device-to-host failed (size: %zu bytes, error: %d)\n", size, err);
    return 1;
  }
  
  /* Check for asynchronous errors */
  err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: Asynchronous CUDA error detected after copy: %d\n", err);
    return 1;
  }
  
  return 0;
#endif
}

int GPUCopyDeviceToDevice(void *dst, const void *src, size_t size)
{
#ifdef GPU_NONE
  memcpy(dst, src, size);
  return 0;
#else
  int err = GPU_MEMCPY(dst, src, size, GPU_MEMCPY_D2D);
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPU copy device-to-device failed (size: %zu bytes)\n", size);
    return 1;
  }
  return 0;
#endif
}

int GPUMemset(void *ptr, int value, size_t size)
{
#ifdef GPU_NONE
  memset(ptr, value, size);
  return 0;
#else
  /* Clear any previous errors */
  GPU_GET_LAST_ERROR();
  
  if (!ptr) {
    fprintf(stderr, "Error: GPUMemset: NULL pointer (size: %zu bytes)\n", size);
    return 1;
  }
  
  int err = GPU_MEMSET(ptr, value, size);
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: GPU memset failed (ptr=%p, size: %zu bytes, error: %d)\n", ptr, size, err);
    return 1;
  }
  
  /* Check for asynchronous errors */
  err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: Asynchronous CUDA error detected after memset (ptr=%p, size: %zu bytes, error: %d)\n", 
            ptr, size, err);
    return 1;
  }
  
  return 0;
#endif
}

void GPUSync(void)
{
#ifndef GPU_NONE
  /* Check for errors before sync */
  int err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: CUDA error detected before sync: %d\n", err);
    /* Don't exit here - let the sync happen and check again */
  }
  
  GPU_DEVICE_SYNC();
  
  /* Check for errors after sync */
  err = GPU_GET_LAST_ERROR();
  if (err != GPU_SUCCESS) {
    fprintf(stderr, "Error: CUDA error detected after sync: %d\n", err);
    exit(1);
  }
#endif
}

GPULaunchConfig GPUConfigureLaunch(size_t n, int blockSize)
{
  GPULaunchConfig config;
  config.blockSize = blockSize;
  config.gridSize = (n + blockSize - 1) / blockSize;
  return config;
}

int GPUIsAvailable(void)
{
#ifdef GPU_NONE
  return 0;
#else
  return 1;
#endif
}

