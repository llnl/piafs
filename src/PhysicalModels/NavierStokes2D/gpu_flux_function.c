/*! @file gpu_flux_function.c
    @brief GPU-enabled NavierStokes2DFlux function
*/

#include <stdio.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_flux.h>
#include <hypar.h>
#include <physicalmodels/navierstokes2d.h>

int GPUNavierStokes2DFlux(double *f, double *u, int dir, void *s, double t)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes2D *param = (NavierStokes2D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUNavierStokes2DFlux: param is NULL\n"); return 1; }
  int ndims = solver->ndims, nvars = param->nvars, ghosts = solver->ghosts;
  int *dim = solver->dim_local, *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;
  gpu_launch_ns2d_flux(f, u, nvars, ndims, dim, stride_with_ghosts, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

