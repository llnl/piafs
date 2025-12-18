/*! @file gpu_flux.h
    @brief GPU flux kernel declarations
*/

#ifndef _GPU_FLUX_H_
#define _GPU_FLUX_H_

#include <gpu.h>

/* Kernel annotations must only be visible to CUDA/HIP compilers. */
#if defined(GPU_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP) && (defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__))
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* NavierStokes3D kernels */
GPU_KERNEL void gpu_ns3d_flux_kernel(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma
);

#ifdef __cplusplus
extern "C" {
#endif

void gpu_launch_ns3d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
);

/* NavierStokes2D kernels */
GPU_KERNEL void gpu_ns2d_flux_kernel(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma
);

void gpu_launch_ns2d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
);

/* Euler1D kernels */
GPU_KERNEL void gpu_euler1d_flux_kernel(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma
);

void gpu_launch_euler1d_flux(
  double *f, const double *u, int nvars, int ndims, const int *dim,
  const int *stride_with_ghosts, int ghosts, int dir, double gamma, int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_FLUX_H_ */
