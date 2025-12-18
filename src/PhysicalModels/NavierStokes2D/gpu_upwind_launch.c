/*! @file gpu_upwind_launch.c
    @brief GPU upwind kernel launch wrappers (CPU fallback) for NS2D
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_upwind.h>
#endif

void gpu_launch_ns2d_upwind_roe(double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize) { (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter; (void)ghosts; (void)dir; (void)gamma; (void)blockSize; }
void gpu_launch_ns2d_upwind_rf(double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize) { (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter; (void)ghosts; (void)dir; (void)gamma; (void)blockSize; }
void gpu_launch_ns2d_upwind_llf(double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize) { (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter; (void)ghosts; (void)dir; (void)gamma; (void)blockSize; }
void gpu_launch_ns2d_upwind_rusanov(double *fI, const double *fL, const double *fR, const double *uL, const double *uR, const double *u,
  int nvars, int ndims, const int *dim, const int *stride_with_ghosts, const int *bounds_inter,
  int ghosts, int dir, double gamma, int blockSize) { (void)fI; (void)fL; (void)fR; (void)uL; (void)uR; (void)u; (void)nvars; (void)ndims; (void)dim; (void)stride_with_ghosts; (void)bounds_inter; (void)ghosts; (void)dir; (void)gamma; (void)blockSize; }

