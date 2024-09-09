/*! @file NavierStokes2DSource.c
    @author Debojyoti Ghosh
    @brief Compute the source term for the 2D Navier Stokes system
*/
#include <stdlib.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <physicalmodels/navierstokes2d.h>
#include <mpivars.h>
#include <hypar.h>

/*! Computes the source term for the 2D Navier-Stokes equations */
int NavierStokes2DSource(
                          double  *source,  /*!< Array to hold the computed source */
                          double  *u,       /*!< Solution vector array */
                          void    *s,       /*!< Solver object of type #HyPar */
                          void    *m,       /*!< MPI object of type #MPIVariables */
                          double  t         /*!< Current simulation time */
                        )
{
  HyPar           *solver = (HyPar* )         s;
  MPIVariables    *mpi    = (MPIVariables*)   m;
  NavierStokes2D  *param  = (NavierStokes2D*) solver->physics;

  _ArraySetValue_(source, _MODEL_NVARS_*solver->npoints_local_wghosts, 0.0);

  if (param->include_chem) {

    Chemistry *chem = (Chemistry*) param->chem;

    int *dim    = solver->dim_local;
    int ghosts  = solver->ghosts;
    int ndims   = solver->ndims;

    int index[ndims];
    int done = 0; _ArraySetValue_(index,ndims,0);
    while (!done) {
      int p; _ArrayIndex1D_(ndims,dim,index,ghosts,p);
      source[_MODEL_NVARS_*p + 3] = chem->Qv[p]/(param->gamma-1.0);
      _ArrayIncrementIndex_(ndims,dim,index,done);
    }

  }

  return(0);
}
