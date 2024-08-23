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

/*! Computes the source term using a well-balanced formulation */
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

  return(0);
}
