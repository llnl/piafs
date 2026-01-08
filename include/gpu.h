/*! @file gpu.h
    @brief GPU abstraction layer for CUDA and HIP
    @author GPU Support Implementation

    This header provides a unified interface for GPU operations that works
    with both CUDA and HIP. All computations and memory allocations happen
    on the device; data is copied to host only for I/O operations.
*/

#ifndef _GPU_H_
#define _GPU_H_

#include <basic.h>
#include <stdlib.h>  /* for exit, malloc, free */
#include <stddef.h>  /* for size_t */
#include <stdio.h>   /* for fprintf, stderr */
#include <string.h>  /* for memcpy, memset */

/* Detect GPU backend - check CMake definitions first */
#ifdef GPU_NONE
  /* GPU explicitly disabled via CMake - do nothing */
#elif defined(GPU_CUDA)
  /* CUDA enabled via CMake */
  /* Only include CUDA headers if we're compiling with nvcc or if __CUDACC__ is defined */
  /* For regular C/C++ files, we'll include cuda_runtime.h only (not cuda.h) */
  #if defined(__CUDACC__) || defined(__CUDA_ARCH__)
    /* Compiling with nvcc - include full CUDA headers */
    #include <cuda.h>
    #include <cuda_runtime.h>
  #else
    /* Compiling with regular compiler - only include runtime (which is C-compatible) */
    #include <cuda_runtime.h>
  #endif
#elif defined(GPU_HIP)
  /* HIP enabled via CMake */
  #if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    /* Compiling with hipcc - include full HIP headers */
    #include <hip/hip_runtime.h>
  #else
    /* Compiling with regular compiler - only include runtime */
    #include <hip/hip_runtime.h>
  #endif
#elif defined(__CUDACC__) || defined(__CUDA_ARCH__)
  /* Auto-detect CUDA (if CMake didn't set anything) */
  #define GPU_CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
#elif defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
  /* Auto-detect HIP (if CMake didn't set anything) */
  #define GPU_HIP
  #include <hip/hip_runtime.h>
#else
  /* No GPU detected */
  #define GPU_NONE
#endif

/* Unified GPU API macros */
#ifdef GPU_CUDA
  #define GPU_MALLOC cudaMalloc
  #define GPU_FREE cudaFree
  #define GPU_MEMCPY cudaMemcpy
  #define GPU_MEMCPY_H2D cudaMemcpyHostToDevice
  #define GPU_MEMCPY_D2H cudaMemcpyDeviceToHost
  #define GPU_MEMCPY_D2D cudaMemcpyDeviceToDevice
  #define GPU_MEMSET cudaMemset
  #define GPU_DEVICE_SYNC cudaDeviceSynchronize
  #define GPU_GET_LAST_ERROR cudaGetLastError
  #define GPU_SUCCESS cudaSuccess
  #define GPU_CHECK_ERROR(x) { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }
  /* CUDA kernel launch - use <<<>>> syntax (only works in .cu files) */
  #define GPU_KERNEL_LAUNCH(kernel, grid, block) kernel<<<grid, block>>>
  #define GPU_KERNEL_END
#elif defined(GPU_HIP)
  #define GPU_MALLOC hipMalloc
  #define GPU_FREE hipFree
  #define GPU_MEMCPY hipMemcpy
  #define GPU_MEMCPY_H2D hipMemcpyHostToDevice
  #define GPU_MEMCPY_D2H hipMemcpyDeviceToHost
  #define GPU_MEMCPY_D2D hipMemcpyDeviceToDevice
  #define GPU_MEMSET hipMemset
  #define GPU_DEVICE_SYNC hipDeviceSynchronize
  #define GPU_GET_LAST_ERROR hipGetLastError
  #define GPU_SUCCESS hipSuccess
  #define GPU_CHECK_ERROR(x) { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      exit(1); \
    } \
  }
  /* HIP kernel launch - support CUDA-like <<<>>> syntax (works with hipcc). */
  #define GPU_KERNEL_LAUNCH(kernel, grid, block) kernel<<<dim3(grid), dim3(block), 0, 0>>>
  #define GPU_KERNEL_END
#else
  /* CPU fallback - allocate on host */
  #define GPU_MALLOC(ptr, size) (*(ptr) = malloc(size), (*(ptr) ? 0 : 1))
  #define GPU_FREE free
  #define GPU_MEMCPY(dst, src, size, kind) memcpy(dst, src, size)
  #define GPU_MEMCPY_H2D 0
  #define GPU_MEMCPY_D2H 1
  #define GPU_MEMCPY_D2D 2
  #define GPU_MEMSET(ptr, val, size) memset(ptr, val, size)
  #define GPU_DEVICE_SYNC() ((void)0)
  #define GPU_GET_LAST_ERROR() (0)
  #define GPU_SUCCESS (0)
  #define GPU_CHECK_ERROR(x) (x)
  #define GPU_KERNEL_LAUNCH(grid, block)
  #define GPU_KERNEL_END
#endif

/* Device pointer type */
#ifdef GPU_CUDA
  typedef double* gpu_ptr;
#elif defined(GPU_HIP)
  typedef double* gpu_ptr;
#else
  typedef double* gpu_ptr;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* GPU memory management functions */
int GPUAllocate(void **ptr, size_t size);
int GPUFree(void *ptr);
int GPUCopyToDevice(void *dst, const void *src, size_t size);
int GPUCopyToHost(void *dst, const void *src, size_t size);
int GPUCopyDeviceToDevice(void *dst, const void *src, size_t size);
int GPUMemset(void *ptr, int value, size_t size);
void GPUSync(void);

/* Pinned (page-locked) host memory for faster CPU-GPU transfers */
int GPUAllocatePinned(void **ptr, size_t size);
int GPUFreePinned(void *ptr);

/* Async memory copy operations (requires pinned host memory for best performance) */
int GPUCopyToDeviceAsync(void *dst, const void *src, size_t size, void *stream);
int GPUCopyToHostAsync(void *dst, const void *src, size_t size, void *stream);

/* GPU kernel launch configuration */
typedef struct {
  int blockSize;
  int gridSize;
} GPULaunchConfig;

GPULaunchConfig GPUConfigureLaunch(size_t n, int blockSize);

/* Check if GPU is available */
int GPUIsAvailable(void);

/* GPU stream management */
int GPUCreateStreams(void **stream_hyp, void **stream_par, void **stream_sou);
int GPUDestroyStreams(void *stream_hyp, void *stream_par, void *stream_sou);
int GPUStreamSynchronize(void *stream);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_H_ */

