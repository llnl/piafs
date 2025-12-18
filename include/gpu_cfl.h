/*! @file gpu_cfl.h
    @brief Header for GPU CFL computation kernels
*/

#ifndef _GPU_CFL_H_
#define _GPU_CFL_H_

/* Kernel annotations must only be visible to CUDA/HIP compilers. */
#if defined(GPU_CUDA) && (defined(__CUDACC__) || defined(__CUDA_ARCH__))
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP) && (defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__))
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* NavierStokes3D CFL kernel */
GPU_KERNEL void gpu_ns3d_compute_cfl_kernel(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  int nvars,
  int ndims,
  const int *dim,
  const int *stride_with_ghosts,
  int ghosts,
  double dt,
  double gamma
);

/* NavierStokes2D CFL kernel */
GPU_KERNEL void gpu_ns2d_compute_cfl_kernel(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  int nvars,
  int ndims,
  const int *dim,
  const int *stride_with_ghosts,
  int ghosts,
  double dt,
  double gamma
);

/* Euler1D CFL kernel */
GPU_KERNEL void gpu_euler1d_compute_cfl_kernel(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  int nvars,
  int ndims,
  const int *dim,
  const int *stride_with_ghosts,
  int ghosts,
  double dt,
  double gamma
);

#ifdef __cplusplus
extern "C" {
#endif

/* Launch wrappers */
int gpu_launch_ns3d_compute_cfl(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  void *s,
  double dt
);

int gpu_launch_ns2d_compute_cfl(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  void *s,
  double dt
);

int gpu_launch_euler1d_compute_cfl(
  const double *u,
  const double *dxinv,
  double *cfl_local,
  void *s,
  double dt
);

/* GPU-enabled ComputeCFL wrapper functions */
double GPUNavierStokes3DComputeCFL(void*, void*, double, double);
double GPUNavierStokes2DComputeCFL(void*, void*, double, double);
double GPUEuler1DComputeCFL(void*, void*, double, double);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_CFL_H_ */

