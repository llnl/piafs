/*! @file gpu_source_function.c
    @brief GPU-enabled NavierStokes2DSource function
*/

#include <stdio.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_source.h>
#include <hypar.h>
#include <physicalmodels/navierstokes2d.h>
#include <physicalmodels/chemistry.h>
#include <physicalmodels/gpu_chemistry.h>

int GPUNavierStokes2DSource(double *source, double *u, void *s, void *m, double t)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes2D *param = (NavierStokes2D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUNavierStokes2DSource: param is NULL\n"); return 1; }
  int nvars = param->nvars, npoints = solver->npoints_local_wghosts;

  // Initialize source to zero
  gpu_launch_ns2d_source_zero(source, nvars, npoints, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();

  // Add chemistry source terms if enabled
  if (param->include_chem) {
    GPUChemistrySource(solver, u, source, param->chem, m, t);
  }

  return 0;
}

