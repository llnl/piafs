/*! @file gpu_upwind_launch.c
    @brief GPU upwind kernel launch wrappers (CPU fallback)
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_upwind.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_ns3d_upwind_roe(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  /* CPU fallback - would need full implementation */
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  /* This should not be called - use .cu version */
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#endif
}

void gpu_launch_ns3d_upwind_rf(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#endif
}

void gpu_launch_ns3d_upwind_llf(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#endif
}

void gpu_launch_ns3d_upwind_rusanov(
  double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize
)
{
#ifdef GPU_NONE
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#else
  (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u;
  (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter;
  (void)ghosts; (void)dir; (void)gamma; (void)blockSize;
#endif
}

