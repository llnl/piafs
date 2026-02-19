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

/* Pinned (page-locked) host memory allocation for faster CPU-GPU transfers */
int GPUAllocatePinned(void **ptr, size_t size)
{
#ifdef GPU_CUDA
  cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: cudaHostAlloc failed (size: %zu bytes): %s\n",
            size, cudaGetErrorString(err));
    return 1;
  }
  return 0;
#elif defined(GPU_HIP)
  hipError_t err = hipHostMalloc(ptr, size, hipHostMallocDefault);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: hipHostMalloc failed (size: %zu bytes): %s\n",
            size, hipGetErrorString(err));
    return 1;
  }
  return 0;
#else
  /* CPU fallback - use regular malloc */
  *ptr = malloc(size);
  if (*ptr == NULL) {
    fprintf(stderr, "Error: malloc failed for pinned memory fallback (size: %zu bytes)\n", size);
    return 1;
  }
  return 0;
#endif
}

int GPUFreePinned(void *ptr)
{
  if (!ptr) return 0;
#ifdef GPU_CUDA
  cudaError_t err = cudaFreeHost(ptr);
  if (err != cudaSuccess) {
    fprintf(stderr, "Warning: cudaFreeHost failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
#elif defined(GPU_HIP)
  hipError_t err = hipHostFree(ptr);
  if (err != hipSuccess) {
    fprintf(stderr, "Warning: hipHostFree failed: %s\n", hipGetErrorString(err));
    return 1;
  }
  return 0;
#else
  free(ptr);
  return 0;
#endif
}

/* Async memory copy operations */
int GPUCopyToDeviceAsync(void *dst, const void *src, size_t size, void *stream)
{
#ifdef GPU_CUDA
  cudaStream_t s = stream ? *(cudaStream_t*)stream : 0;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, s);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: cudaMemcpyAsync H2D failed (size: %zu bytes): %s\n",
            size, cudaGetErrorString(err));
    return 1;
  }
  return 0;
#elif defined(GPU_HIP)
  hipStream_t s = stream ? *(hipStream_t*)stream : 0;
  hipError_t err = hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, s);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: hipMemcpyAsync H2D failed (size: %zu bytes): %s\n",
            size, hipGetErrorString(err));
    return 1;
  }
  return 0;
#else
  (void)stream;
  memcpy(dst, src, size);
  return 0;
#endif
}

int GPUCopyToHostAsync(void *dst, const void *src, size_t size, void *stream)
{
#ifdef GPU_CUDA
  cudaStream_t s = stream ? *(cudaStream_t*)stream : 0;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, s);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: cudaMemcpyAsync D2H failed (size: %zu bytes): %s\n",
            size, cudaGetErrorString(err));
    return 1;
  }
  return 0;
#elif defined(GPU_HIP)
  hipStream_t s = stream ? *(hipStream_t*)stream : 0;
  hipError_t err = hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, s);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: hipMemcpyAsync D2H failed (size: %zu bytes): %s\n",
            size, hipGetErrorString(err));
    return 1;
  }
  return 0;
#else
  (void)stream;
  memcpy(dst, src, size);
  return 0;
#endif
}

int GPUIsAvailable(void)
{
#ifdef GPU_NONE
  return 0;
#else
  return 1;
#endif
}

int GPUCreateStreams(void **stream_hyp, void **stream_par, void **stream_sou)
{
#ifdef GPU_CUDA
  cudaStream_t *s_hyp = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStream_t *s_par = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStream_t *s_sou = (cudaStream_t*)malloc(sizeof(cudaStream_t));

  if (!s_hyp || !s_par || !s_sou) {
    fprintf(stderr, "Error: Failed to allocate memory for CUDA streams\n");
    if (s_hyp) free(s_hyp);
    if (s_par) free(s_par);
    if (s_sou) free(s_sou);
    return 1;
  }

  cudaError_t err;
  err = cudaStreamCreate(s_hyp);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to create CUDA stream (hyp): %s\n", cudaGetErrorString(err));
    free(s_hyp); free(s_par); free(s_sou);
    return 1;
  }

  err = cudaStreamCreate(s_par);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to create CUDA stream (par): %s\n", cudaGetErrorString(err));
    cudaStreamDestroy(*s_hyp);
    free(s_hyp); free(s_par); free(s_sou);
    return 1;
  }

  err = cudaStreamCreate(s_sou);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to create CUDA stream (sou): %s\n", cudaGetErrorString(err));
    cudaStreamDestroy(*s_hyp);
    cudaStreamDestroy(*s_par);
    free(s_hyp); free(s_par); free(s_sou);
    return 1;
  }

  *stream_hyp = (void*)s_hyp;
  *stream_par = (void*)s_par;
  *stream_sou = (void*)s_sou;
  return 0;

#elif defined(GPU_HIP)
  hipStream_t *s_hyp = (hipStream_t*)malloc(sizeof(hipStream_t));
  hipStream_t *s_par = (hipStream_t*)malloc(sizeof(hipStream_t));
  hipStream_t *s_sou = (hipStream_t*)malloc(sizeof(hipStream_t));

  if (!s_hyp || !s_par || !s_sou) {
    fprintf(stderr, "Error: Failed to allocate memory for HIP streams\n");
    if (s_hyp) free(s_hyp);
    if (s_par) free(s_par);
    if (s_sou) free(s_sou);
    return 1;
  }

  hipError_t err;
  err = hipStreamCreate(s_hyp);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: Failed to create HIP stream (hyp): %s\n", hipGetErrorString(err));
    free(s_hyp); free(s_par); free(s_sou);
    return 1;
  }

  err = hipStreamCreate(s_par);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: Failed to create HIP stream (par): %s\n", hipGetErrorString(err));
    hipStreamDestroy(*s_hyp);
    free(s_hyp); free(s_par); free(s_sou);
    return 1;
  }

  err = hipStreamCreate(s_sou);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: Failed to create HIP stream (sou): %s\n", hipGetErrorString(err));
    hipStreamDestroy(*s_hyp);
    hipStreamDestroy(*s_par);
    free(s_hyp); free(s_par); free(s_sou);
    return 1;
  }

  *stream_hyp = (void*)s_hyp;
  *stream_par = (void*)s_par;
  *stream_sou = (void*)s_sou;
  return 0;

#else
  *stream_hyp = NULL;
  *stream_par = NULL;
  *stream_sou = NULL;
  return 0;
#endif
}

int GPUDestroyStreams(void *stream_hyp, void *stream_par, void *stream_sou)
{
#ifdef GPU_CUDA
  if (stream_hyp) {
    cudaStreamDestroy(*(cudaStream_t*)stream_hyp);
    free(stream_hyp);
  }
  if (stream_par) {
    cudaStreamDestroy(*(cudaStream_t*)stream_par);
    free(stream_par);
  }
  if (stream_sou) {
    cudaStreamDestroy(*(cudaStream_t*)stream_sou);
    free(stream_sou);
  }
  return 0;

#elif defined(GPU_HIP)
  if (stream_hyp) {
    hipStreamDestroy(*(hipStream_t*)stream_hyp);
    free(stream_hyp);
  }
  if (stream_par) {
    hipStreamDestroy(*(hipStream_t*)stream_par);
    free(stream_par);
  }
  if (stream_sou) {
    hipStreamDestroy(*(hipStream_t*)stream_sou);
    free(stream_sou);
  }
  return 0;

#else
  (void)stream_hyp;
  (void)stream_par;
  (void)stream_sou;
  return 0;
#endif
}

int GPUStreamSynchronize(void *stream)
{
#ifdef GPU_CUDA
  if (stream) {
    cudaError_t err = cudaStreamSynchronize(*(cudaStream_t*)stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "Error: CUDA stream synchronization failed: %s\n", cudaGetErrorString(err));
      return 1;
    }
  }
  return 0;

#elif defined(GPU_HIP)
  if (stream) {
    hipError_t err = hipStreamSynchronize(*(hipStream_t*)stream);
    if (err != hipSuccess) {
      fprintf(stderr, "Error: HIP stream synchronization failed: %s\n", hipGetErrorString(err));
      return 1;
    }
  }
  return 0;

#else
  (void)stream;
  return 0;
#endif
}

