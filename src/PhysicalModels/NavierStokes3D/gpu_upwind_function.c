/*! @file gpu_upwind_function.c
    @brief GPU-enabled NavierStokes3DUpwindRoe function
*/

#include <stdio.h>
#include <stdlib.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_upwind.h>
#include <hypar.h>
#include <physicalmodels/navierstokes3d.h>

/* GPU version of NavierStokes3DUpwindRoe */
int GPUNavierStokes3DUpwindRoe(
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
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  
  if (!param) {
    fprintf(stderr, "Error: GPUNavierStokes3DUpwindRoe: param is NULL\n");
    return 1;
  }
  
  int ndims = solver->ndims;
  int nvars = param->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  
  /* Compute bounds for interface array */
  int bounds_inter[3]; /* Support up to 3D */
  for (int i = 0; i < ndims; i++) {
    bounds_inter[i] = dim[i];
  }
  bounds_inter[dir] = dim[dir] + 1; /* One more interface than cells */
  
  /* Launch GPU kernel */
  gpu_launch_ns3d_upwind_roe(
    fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, gamma, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  return 0;
}

int GPUNavierStokes3DUpwindRF(
  double *fI, double *fL, double *fR, double *uL, double *uR, double *u,
  int dir, void *s, double t
)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUNavierStokes3DUpwindRF: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  gpu_launch_ns3d_upwind_rf(fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

int GPUNavierStokes3DUpwindLLF(
  double *fI, double *fL, double *fR, double *uL, double *uR, double *u,
  int dir, void *s, double t
)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUNavierStokes3DUpwindLLF: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  gpu_launch_ns3d_upwind_llf(fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

int GPUNavierStokes3DUpwindRusanov(
  double *fI, double *fL, double *fR, double *uL, double *uR, double *u,
  int dir, void *s, double t
)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUNavierStokes3DUpwindRusanov: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  gpu_launch_ns3d_upwind_rusanov(fI, fL, fR, uL, uR, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

