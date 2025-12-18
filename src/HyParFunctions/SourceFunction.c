/*! @file SourceFunction.c
    @author Debojyoti Ghosh
    @brief Evaluate the source term
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <boundaryconditions.h>
#include <mpivars.h>
#include <hypar.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#endif

/*! Evaluate the source term \f${\bf S}\left({\bf u}\right)\f$ in the governing equation,
    if the physical model specifies one. In addition, if the simulation requires a sponge
    boundary treatment, the sponge BC function is called.
*/
int SourceFunction(
                    double  *source,  /*!< the computed source term */
                    double  *u,       /*!< solution */
                    void    *s,       /*!< solver object of type #HyPar */
                    void    *m,       /*!< MPI object of type #MPIVariables */
                    double  t         /*!< Current simulation time */
                  )
{
  HyPar           *solver   = (HyPar*)        s;
  MPIVariables    *mpi      = (MPIVariables*) m;

  /* initialize to zero */
  int size = solver->ndof_cells_wghosts;
#ifdef GPU_CUDA
  if (GPUShouldUse()) {
    GPUArraySetValue(source, 0.0, size);
  } else {
    _ArraySetValue_(source,size,0.0);
  }
#elif defined(GPU_HIP)
  if (GPUShouldUse()) {
    GPUArraySetValue(source, 0.0, size);
  } else {
    _ArraySetValue_(source,size,0.0);
  }
#else
  _ArraySetValue_(source,size,0.0);
#endif

  /* call the source function of the physics model, if available */
  if (solver->SFunction) {
    solver->SFunction(source,u,solver,mpi,t);
    solver->count_sou++;
  }

  /* Apart from other source terms, implement sponge BC as a source */
  DomainBoundary* boundary = (DomainBoundary*) solver->boundary;
  int n;
  int nb = solver->nBoundaryZones;
  for (n = 0; n < nb; n++) {
    if (!strcmp(boundary[n].bctype,_SPONGE_)) {
      /* x is always on host, no GPU copy needed */
      BCSpongeSource( &boundary[n],
                      solver->ndims,
                      solver->nvars,
                      solver->ghosts,
                      solver->dim_local,
                      solver->x,
                      u,
                      source  );
    }
  }

  return(0);
}
