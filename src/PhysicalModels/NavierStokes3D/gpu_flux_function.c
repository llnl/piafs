/*! @file gpu_flux_function.c
    @brief GPU-enabled NavierStokes3DFlux function
*/

#include <stdio.h>
#include <stdlib.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_flux.h>
#include <hypar.h>
#include <physicalmodels/navierstokes3d.h>

/* GPU version of NavierStokes3DFlux */
int GPUNavierStokes3DFlux(
  double *f,              /* output: flux array */
  double *u,              /* input: solution array */
  int dir,                /* direction */
  void *s,                /* solver object */
  double t                /* time */
)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;

  if (!param) {
    fprintf(stderr, "Error: GPUNavierStokes3DFlux: param is NULL\n");
    return 1;
  }

  int ndims = solver->ndims;
  int nvars = param->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  double gamma = param->gamma;

  /* Launch GPU kernel */
  gpu_launch_ns3d_flux(f, u, nvars, ndims, dim, stride_with_ghosts, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();

  return 0;
}

