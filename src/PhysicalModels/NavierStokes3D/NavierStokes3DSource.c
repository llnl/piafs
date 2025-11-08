/*! @file NavierStokes3DSource.c
    @author Debojyoti Ghosh
    @brief Compute the source term for the 3D Navier Stokes system
*/
#include <stdlib.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <physicalmodels/navierstokes3d.h>
#include <mpivars.h>
#include <hypar.h>

/*! Computes the source term for the 3D Navier-Stokes equations */
int NavierStokes3DSource( double  *source,  /*!< Array to hold the computed source */
                          double  *u,       /*!< Solution vector array */
                          void    *s,       /*!< Solver object of type #HyPar */
                          void    *m,       /*!< MPI object of type #MPIVariables */
                          double  t         /*!< Current simulation time */ )
{
  HyPar           *solver = (HyPar* )         s;
  MPIVariables    *mpi    = (MPIVariables*)   m;
  NavierStokes3D  *param  = (NavierStokes3D*) solver->physics;

  _ArraySetValue_(source, param->nvars*solver->npoints_local_wghosts, 0.0);

  if (param->include_chem) {
    Chemistry *chem = (Chemistry*) param->chem;
    ChemistrySource(solver, u, source, chem, mpi, t);
  }

  return(0);
}
