/*! @file gpu_source_function.c
    @brief GPU-enabled NavierStokes3DSource function
*/

#include <stdio.h>
#include <stdlib.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_source.h>
#include <hypar.h>
#include <physicalmodels/navierstokes3d.h>
#include <physicalmodels/chemistry.h>
#include <physicalmodels/gpu_chemistry.h>

/* GPU version of NavierStokes3DSource */
int GPUNavierStokes3DSource(
  double *source,         /* output: source array */
  double *u,              /* input: solution array */
  void *s,                /* solver object */
  void *m,                /* MPI object */
  double t                /* time */
)
{
  HyPar *solver = (HyPar*) s;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  
  if (!param) {
    fprintf(stderr, "Error: GPUNavierStokes3DSource: param is NULL\n");
    return 1;
  }
  
  int nvars = param->nvars;
  int npoints = solver->npoints_local_wghosts;
  
  /* Set source to zero on GPU */
  gpu_launch_ns3d_source_zero(source, nvars, npoints, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  /* Add chemistry source terms if enabled */
  if (param->include_chem) {
    GPUChemistrySource(solver, u, source, param->chem, m, t);
  }
  
  return 0;
}

