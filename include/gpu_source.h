/*! @file gpu_source.h
    @brief GPU source kernel declarations
*/

#ifndef _GPU_SOURCE_H_
#define _GPU_SOURCE_H_

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
GPU_KERNEL void gpu_ns3d_source_zero_kernel(
  double *source, int nvars, int npoints
);

GPU_KERNEL void gpu_ns2d_source_zero_kernel(double *source, int nvars, int npoints);
GPU_KERNEL void gpu_euler1d_source_zero_kernel(double *source, int nvars, int npoints);

/* Launch wrappers */
#ifdef __cplusplus
extern "C" {
#endif

void gpu_launch_ns3d_source_zero(double *source, int nvars, int npoints, int blockSize);
void gpu_launch_ns2d_source_zero(double *source, int nvars, int npoints, int blockSize);
void gpu_launch_euler1d_source_zero(double *source, int nvars, int npoints, int blockSize);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_SOURCE_H_ */

