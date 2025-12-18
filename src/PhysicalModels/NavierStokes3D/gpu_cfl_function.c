/*! @file gpu_cfl_function.c
    @brief GPU-enabled NavierStokes3DComputeCFL function
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_cfl.h>
#include <gpu_launch.h>
#include <physicalmodels/navierstokes3d.h>
#include <hypar.h>
#include <mpivars.h>
#include <stdlib.h>
#include <math.h>

/* GPU-enabled ComputeCFL */
double GPUNavierStokes3DComputeCFL(
  void *s,
  void *m,
  double dt,
  double t
)
{
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  NavierStokes3D *param = (NavierStokes3D*) solver->physics;
  
  if (!GPUShouldUse()) {
    /* Fall back to CPU version */
    extern double NavierStokes3DComputeCFL(void*, void*, double, double);
    return NavierStokes3DComputeCFL(s, m, dt, t);
  }
  
  int ndims = solver->ndims;
  int *dim = solver->dim_local;
  
  /* Compute total number of grid points (without ghosts) */
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }
  
  if (npoints == 0) {
    return 0.0;
  }
  
  /* Allocate GPU array for local CFL values */
  double *cfl_local_gpu = NULL;
  if (GPUAllocate((void**)&cfl_local_gpu, npoints * sizeof(double))) {
    fprintf(stderr, "Error: Failed to allocate GPU memory for CFL computation\n");
    return -1.0;
  }
  
  /* Launch GPU kernel */
  if (gpu_launch_ns3d_compute_cfl(
    solver->u,
    solver->d_dxinv,  /* Use device copy of dxinv */
    cfl_local_gpu,
    solver,
    dt
  )) {
    GPUFree(cfl_local_gpu);
    fprintf(stderr, "Error: GPU CFL kernel launch failed\n");
    return -1.0;
  }
  
  /* Find maximum CFL using GPU reduction */
  double max_cfl = gpu_launch_array_max(cfl_local_gpu, npoints, 256);
  GPUFree(cfl_local_gpu);
  
  /* Get global maximum across MPI ranks */
  double global_max_cfl = 0.0;
  MPIMax_double(&global_max_cfl, &max_cfl, 1, &mpi->world);
  
  return global_max_cfl;
}

