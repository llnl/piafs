/*! @file CalculateError.c
    @author Debojyoti Ghosh
    @brief Computes the error in the solution.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <basic.h>
#include <common.h>
#include <arrayfunctions.h>
#include <timeintegration.h>
#include <mpivars.h>
#include <hypar.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#endif

int ExactSolution(void*,void*,double*,char*,int*);

/*! Calculates the error in the solution if the exact solution is
    available. If the exact solution is not available, the errors
    are reported as zero.
    The exact solution should be provided in the file "exact.inp"
    in the same format as the initial solution.
*/
int CalculateError(
                    void *s, /*!< Solver object of type #HyPar */
                    void *m  /*!< MPI object of type #MPIVariables */
                  )
{
  HyPar         *solver     = (HyPar*)        s;
  MPIVariables  *mpi        = (MPIVariables*) m;
  int           exact_flag  = 0, i, size;
  double        sum         = 0, global_sum = 0;
  double        *uex        = NULL;
  _DECLARE_IERR_;

  size = solver->nvars;
  for (i = 0; i < solver->ndims; i++)
    size *= (solver->dim_local[i]+2*solver->ghosts);
  uex = (double*) calloc (size, sizeof(double));

  char fname_root[_MAX_STRING_SIZE_] = "exact";
  if (solver->nsims > 1) {
    char index[_MAX_STRING_SIZE_];
    GetStringFromInteger(solver->my_idx, index, (int)log10(solver->nsims)+1);
    strcat(fname_root, "_");
    strcat(fname_root, index);
  }

  static const double tolerance = 1e-15;
  IERR ExactSolution( solver,
                      mpi,
                      uex,
                      fname_root,
                      &exact_flag ); CHECKERR(ierr);

  if (!exact_flag) {

    /* No exact solution */
    solver->error[0] = solver->error[1] = solver->error[2] = -1;

  } else {

    /* calculate solution norms (for relative error) */
    double solution_norm[3] = {0.0,0.0,0.0};
    /* L1 */
    sum = ArraySumAbsnD   (solver->nvars,solver->ndims,solver->dim_local,
                           solver->ghosts,solver->index,uex);
    global_sum = 0; MPISum_double(&global_sum,&sum,1,&mpi->world);
    solution_norm[0] = global_sum/((double)solver->npoints_global);
    /* L2 */
    sum = ArraySumSquarenD(solver->nvars,solver->ndims,solver->dim_local,
                           solver->ghosts,solver->index,uex);
    global_sum = 0; MPISum_double(&global_sum,&sum,1,&mpi->world);
    if (solver->npoints_global == 0) {
      fprintf(stderr,"ERROR in CalculateError: npoints_global is zero, cannot compute norm.\n");
      exit(1);
    }
    solution_norm[1] = sqrt(global_sum/((double)solver->npoints_global));
    if (isnan(solution_norm[1]) || isinf(solution_norm[1])) {
      fprintf(stderr,"ERROR in CalculateError: NaN/Inf in L2 solution norm.\n");
      exit(1);
    }
    /* Linf */
    sum = ArrayMaxnD      (solver->nvars,solver->ndims,solver->dim_local,
                           solver->ghosts,solver->index,uex);
    global_sum = 0; MPIMax_double(&global_sum,&sum,1,&mpi->world);
    solution_norm[2] = global_sum;

    /* compute error = difference between exact and numerical solution */
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      /* Copy solver->u to host for computation */
      double *u_host = (double*) malloc(size * sizeof(double));
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for CalculateError\n");
        free(uex);
        return 1;
      }
      GPUCopyToHost(u_host, solver->u, size * sizeof(double));
      /* GPUCopyToHost uses a synchronous copy; avoid forced device sync here. */
      _ArrayAXPY_(u_host,-1.0,uex,size);
      free(u_host);
    } else {
      _ArrayAXPY_(solver->u,-1.0,uex,size);
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      /* Copy solver->u to host for computation */
      double *u_host = (double*) malloc(size * sizeof(double));
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for CalculateError\n");
        free(uex);
        return 1;
      }
      GPUCopyToHost(u_host, solver->u, size * sizeof(double));
      /* GPUCopyToHost uses a synchronous copy; avoid forced device sync here. */
      _ArrayAXPY_(u_host,-1.0,uex,size);
      free(u_host);
    } else {
      _ArrayAXPY_(solver->u,-1.0,uex,size);
    }
#else
    _ArrayAXPY_(solver->u,-1.0,uex,size);
#endif

    /* calculate L1 norm of error */
    sum = ArraySumAbsnD   (solver->nvars,solver->ndims,solver->dim_local,
                           solver->ghosts,solver->index,uex);
    global_sum = 0; MPISum_double(&global_sum,&sum,1,&mpi->world);
    solver->error[0] = global_sum/((double)solver->npoints_global);

    /* calculate L2 norm of error */
    sum = ArraySumSquarenD(solver->nvars,solver->ndims,solver->dim_local,
                           solver->ghosts,solver->index,uex);
    global_sum = 0; MPISum_double(&global_sum,&sum,1,&mpi->world);
    solver->error[1] = sqrt(global_sum/((double)solver->npoints_global));
    if (isnan(solver->error[1]) || isinf(solver->error[1])) {
      fprintf(stderr,"ERROR in CalculateError: NaN/Inf in L2 error norm.\n");
      exit(1);
    }

    /* calculate Linf norm of error */
    sum = ArrayMaxnD      (solver->nvars,solver->ndims,solver->dim_local,
                           solver->ghosts,solver->index,uex);
    global_sum = 0; MPIMax_double(&global_sum,&sum,1,&mpi->world);
    solver->error[2] = global_sum;
    if (isnan(solver->error[2]) || isinf(solver->error[2])) {
      fprintf(stderr,"ERROR in CalculateError: NaN/Inf in Linf error norm.\n");
      exit(1);
    }

    /*
      decide whether to normalize and report relative errors,
      or report absolute errors.
    */
    if (    (solution_norm[0] > tolerance)
        &&  (solution_norm[1] > tolerance)
        &&  (solution_norm[2] > tolerance) ) {
      solver->error[0] /= solution_norm[0];
      solver->error[1] /= solution_norm[1];
      solver->error[2] /= solution_norm[2];
    }
  }

  free(uex);
  return(0);
}

