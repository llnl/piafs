/*! @file gpu_interpolation.h
    @brief GPU interpolation kernel declarations
*/

#ifndef _GPU_INTERPOLATION_H_
#define _GPU_INTERPOLATION_H_

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
GPU_KERNEL void gpu_weno5_interpolation_kernel(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ninterfaces, int stride, int upw
);

GPU_KERNEL void gpu_central2_interpolation_kernel(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride
);

GPU_KERNEL void gpu_central4_interpolation_kernel(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride
);

GPU_KERNEL void gpu_muscl3_interpolation_kernel(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int upw, double eps
);

/* Launch wrappers */
#ifdef __cplusplus
extern "C" {
#endif

void gpu_launch_weno5_interpolation(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ninterfaces, int stride, int upw, int blockSize
);

void gpu_launch_central2_interpolation(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int blockSize
);

void gpu_launch_central4_interpolation(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int blockSize
);

void gpu_launch_muscl3_interpolation(
  double *fI, const double *fC, int nvars, int ninterfaces, int stride, int upw, double eps, int blockSize
);

/* Multi-dimensional WENO5 interpolation */
GPU_KERNEL void gpu_weno5_interpolation_nd_kernel(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw
);

void gpu_launch_weno5_interpolation_nd(
  double *fI, const double *fC, const double *w1, const double *w2, const double *w3,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, int blockSize
);

/* Multi-dimensional WENO5 characteristic-based interpolation */
GPU_KERNEL void gpu_weno5_interpolation_nd_char_kernel(
  double *fI, const double *fC, const double *u, const double *w1, const double *w2, const double *w3,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma
);

void gpu_launch_weno5_interpolation_nd_char(
  double *fI, const double *fC, const double *u, const double *w1, const double *w2, const double *w3,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma, int blockSize
);

/* Multi-dimensional MUSCL2 interpolation (component-wise) */
GPU_KERNEL void gpu_muscl2_interpolation_nd_kernel(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id
);

void gpu_launch_muscl2_interpolation_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id,
  int blockSize
);

/* Multi-dimensional MUSCL3 interpolation (component-wise, Koren limiter form used by CPU code) */
GPU_KERNEL void gpu_muscl3_interpolation_nd_kernel(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps
);

void gpu_launch_muscl3_interpolation_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps,
  int blockSize
);

/* Multi-dimensional MUSCL2 characteristic interpolation (NS3D: first 5 vars in characteristic space, passives component-wise) */
GPU_KERNEL void gpu_muscl2_interpolation_nd_char_ns3d_kernel(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id,
  double gamma
);

void gpu_launch_muscl2_interpolation_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  int limiter_id,
  double gamma,
  int blockSize
);

/* Multi-dimensional MUSCL3 characteristic interpolation (NS3D: first 5 vars in characteristic space, passives component-wise) */
GPU_KERNEL void gpu_muscl3_interpolation_nd_char_ns3d_kernel(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps,
  double gamma
);

void gpu_launch_muscl3_interpolation_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw,
  double eps,
  double gamma,
  int blockSize
);

/* First order upwind interpolation (component-wise) */
void gpu_launch_first_order_upwind_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, int blockSize
);

/* Second order central interpolation (component-wise) */
void gpu_launch_second_order_central_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int blockSize
);

/* Fourth order central interpolation (component-wise) */
void gpu_launch_fourth_order_central_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int blockSize
);

/* Fifth order upwind interpolation (component-wise) */
void gpu_launch_fifth_order_upwind_nd(
  double *fI, const double *fC,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, int blockSize
);

/* Characteristic versions (NS3D) */
void gpu_launch_first_order_upwind_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma, int blockSize
);

void gpu_launch_second_order_central_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
);

void gpu_launch_fourth_order_central_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
);

void gpu_launch_fifth_order_upwind_nd_char_ns3d(
  double *fI, const double *fC, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, int upw, double gamma, int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_INTERPOLATION_H_ */

