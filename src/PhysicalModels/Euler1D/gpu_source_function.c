/*! @file gpu_source_function.c
    @brief GPU-enabled Euler1DSource function
*/

#include <stdio.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_source.h>
#include <hypar.h>
#include <physicalmodels/euler1d.h>
#include <physicalmodels/chemistry.h>
#include <physicalmodels/gpu_chemistry.h>

int GPUEuler1DSource(double *source, double *u, void *s, void *m, double t)
{
  fflush(stderr);
  
  HyPar *solver = (HyPar*) s;
  Euler1D *param = (Euler1D*) solver->physics;
  if (!param) { fprintf(stderr, "Error: GPUEuler1DSource: param is NULL\n"); return 1; }
  int nvars = param->nvars, npoints = solver->npoints_local_wghosts;
  
  fflush(stderr);
  
  // Initialize source to zero
  gpu_launch_euler1d_source_zero(source, nvars, npoints, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  fflush(stderr);
  
  // Add chemistry source terms if enabled
  if (param->include_chem) {
    fflush(stderr);
    GPUChemistrySource(solver, u, source, param->chem, m, t);
    fflush(stderr);
  }
  
  fflush(stderr);
  
  return 0;
}

