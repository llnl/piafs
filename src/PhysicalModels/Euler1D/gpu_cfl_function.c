/*! @file gpu_cfl_function.c
    @brief GPU-enabled Euler1DComputeCFL function
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_cfl.h>
#include <gpu_launch.h>
#include <physicalmodels/euler1d.h>
#include <hypar.h>
#include <mpivars.h>
#include <stdlib.h>
#include <math.h>

double GPUEuler1DComputeCFL(
  void *s,
  void *m,
  double dt,
  double t
)
{
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  Euler1D *param = (Euler1D*) solver->physics;
  
  if (!GPUShouldUse()) {
    extern double Euler1DComputeCFL(void*, void*, double, double);
    return Euler1DComputeCFL(s, m, dt, t);
  }
  
  int ndims = solver->ndims;
  int *dim = solver->dim_local;
  
  int npoints = 1;
  for (int i = 0; i < ndims; i++) {
    npoints *= dim[i];
  }
  
  if (npoints == 0) {
    return 0.0;
  }

  /* Use persistent CFL workspace buffer */
  double *cfl_local_gpu = solver->gpu_cfl_workspace;
  if (!cfl_local_gpu || solver->gpu_cfl_workspace_size < (size_t)npoints) {
    fprintf(stderr, "Error: CFL workspace not allocated or too small (%zu < %d)\n",
            solver->gpu_cfl_workspace_size, npoints);
    return -1.0;
  }

  if (gpu_launch_euler1d_compute_cfl(
    solver->u,
    solver->d_dxinv,  /* Use device copy of dxinv */
    cfl_local_gpu,
    solver,
    dt
  )) {
    fprintf(stderr, "Error: GPU CFL kernel launch failed\n");
    return -1.0;
  }

  /* Find maximum CFL using GPU reduction with persistent buffers */
  double max_cfl = gpu_launch_array_max_opt(cfl_local_gpu, npoints,
                                              solver->gpu_reduce_buffer,
                                              solver->gpu_reduce_buffer_size,
                                              solver->gpu_reduce_result, 256);
  
  double global_max_cfl = 0.0;
  MPIMax_double(&global_max_cfl, &max_cfl, 1, &mpi->world);
  
  return global_max_cfl;
}

