/*! @file NavierStokes2DPreStep.c
    @author Debojyoti Ghosh
    @brief Contains the 2D Navier-Stokes-specific function to be called at the beginning of each time step.
*/

#include <physicalmodels/navierstokes2d.h>
#include <hypar.h>

/*! 2D Navier-Stokes-specific function called at the beginning of each time-step:
    Solve and update the ceoncentration of reacting species and UV photons
*/
int NavierStokes2DPreStep( double  *u,   /*!< Solution (conserved variables) */
                           void    *s,   /*!< Solver object of type #HyPar */
                           void    *m,   /*!< MPI object of type #MPIVariables */
                           double  dt,   /*!< Time step size */
                           double  waqt  /*!< Current solution time */ )
{
  HyPar* solver = (HyPar*) s;
  NavierStokes2D* param = (NavierStokes2D*) solver->physics;

  if (param->include_chem) ChemistrySolve(s, param->chem, m, dt, waqt);

  return 0;
}
