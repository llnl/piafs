/*! @file gpu_hyperbolic.h
    @brief GPU hyperbolic function declarations
*/

#ifndef _GPU_HYPERBOLIC_H_
#define _GPU_HYPERBOLIC_H_

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
GPU_KERNEL void gpu_hyperbolic_flux_derivative_kernel(
  double *hyp, const double *fluxI, const double *dxinv,
  int nvars, int npoints, int dir_offset
);

GPU_KERNEL void gpu_hyperbolic_flux_derivative_nd_kernel(
  double *hyp, const double *fluxI, const double *dxinv,
  double *StageBoundaryIntegral,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int dir_offset
);

GPU_KERNEL void gpu_default_upwinding_kernel(
  double *fI, const double *fL, const double *fR,
  int nvars, int ninterfaces
);

/* Kernel launch functions */
#ifdef __cplusplus
extern "C" {
#endif

void gpu_launch_hyperbolic_flux_derivative(
  double *hyp, const double *fluxI, const double *dxinv,
  int nvars, int npoints, int dir_offset, int blockSize
);

void gpu_launch_hyperbolic_flux_derivative_nd(
  double *hyp, const double *fluxI, const double *dxinv,
  double *StageBoundaryIntegral,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int dir_offset, int blockSize
);

void gpu_launch_default_upwinding(
  double *fI, const double *fL, const double *fR,
  int nvars, int ninterfaces, int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_HYPERBOLIC_H_ */

