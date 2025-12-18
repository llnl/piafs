/*! @file gpu_upwind_function.c
    @brief GPU-enabled Euler1D upwind functions
*/

#include <stdio.h>
#include <stdlib.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_upwind.h>
#include <hypar.h>
#include <physicalmodels/euler1d.h>

/* GPU version of Euler1DUpwindRoe */
int GPUEuler1DUpwindRoe(
  double *fI,              /* output: upwind interface flux */
  double *fL,              /* input: left-biased flux */
  double *fR,              /* input: right-biased flux */
  double *uL,              /* input: left-biased solution */
  double *uR,              /* input: right-biased solution */
  double *u,               /* input: cell-centered solution */
  int dir,                 /* direction */
  void *s,                 /* solver object */
  double t                 /* time */
)
{
  HyPar *solver = (HyPar*) s;
  Euler1D *param = (Euler1D*) solver->physics;

  if (!param) {
    fprintf(stderr, "Error: GPUEuler1DUpwindRoe: param is NULL\n");
    return 1;
  }

  int ndims = solver->ndims;
  int nvars = param->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;

  /* Compute bounds for interface array */
  int bounds_inter[1];
  bounds_inter[0] = dim[0] + 1;

  /* Launch GPU kernel */
  gpu_launch_euler1d_upwind_roe(
    fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, gamma, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();

  return 0;
}

/* GPU version of Euler1DUpwindRF */
int GPUEuler1DUpwindRF(
  double *fI, double *fL, double *fR, double *uL, double *uR, double *u,
  int dir, void *s, double t
)
{
  HyPar *solver = (HyPar*) s;
  Euler1D *param = (Euler1D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUEuler1DUpwindRF: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  int bounds_inter[1];
  bounds_inter[0] = dim[0] + 1;
  gpu_launch_euler1d_upwind_rf(fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU version of Euler1DUpwindLLF */
int GPUEuler1DUpwindLLF(
  double *fI, double *fL, double *fR, double *uL, double *uR, double *u,
  int dir, void *s, double t
)
{
  HyPar *solver = (HyPar*) s;
  Euler1D *param = (Euler1D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUEuler1DUpwindLLF: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  int bounds_inter[1];
  bounds_inter[0] = dim[0] + 1;
  gpu_launch_euler1d_upwind_llf(fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU version of Euler1DUpwindRusanov */
int GPUEuler1DUpwindRusanov(
  double *fI, double *fL, double *fR, double *uL, double *uR, double *u,
  int dir, void *s, double t
)
{
  HyPar *solver = (HyPar*) s;
  Euler1D *param = (Euler1D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUEuler1DUpwindRusanov: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  int bounds_inter[1];
  bounds_inter[0] = dim[0] + 1;
  gpu_launch_euler1d_upwind_rusanov(fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}
