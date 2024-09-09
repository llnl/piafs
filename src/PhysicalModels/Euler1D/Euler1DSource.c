/*! @file Euler1DSource.c
    @author Debojyoti Ghosh
    @brief Contains the functions to compute the source terms for the 1D Euler equations.
*/

#include <stdlib.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <physicalmodels/euler1d.h>
#include <mpivars.h>
#include <hypar.h>

/*! Compute the source terms for the 1D Euler equations. */
int Euler1DSource(
                  double  *source, /*!< Computed source terms (array size & layout same as u) */
                  double  *u,      /*!< Solution (conserved variables) */
                  void    *s,      /*!< Solver object of type #HyPar */
                  void    *m,      /*!< MPI object of type #MPIVariables */
                  double  t        /*!< Current solution time */
                 )
{
  HyPar         *solver = (HyPar* ) s;
  MPIVariables  *mpi = (MPIVariables*) m;
  Euler1D       *param  = (Euler1D*) solver->physics;

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
      double rho, v, e, P, c, dxinv, local_cfl;
      _Euler1DGetFlowVar_((u+_MODEL_NVARS_*p),rho,v,e,P,param);

      source[_MODEL_NVARS_*p + 2] = chem->Qv[p]/(param->gamma-1.0);

      _ArrayIncrementIndex_(ndims,dim,index,done);
    }

  }

  return 0;
}
