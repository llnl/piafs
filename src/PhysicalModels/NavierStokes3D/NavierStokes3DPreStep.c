/*! @file NavierStokes3DPreStep.c
    @author Debojyoti Ghosh
    @brief Contains the 3D Navier-Stokes-specific function to be called at the beginning of each time step.
*/

#include <physicalmodels/navierstokes3d.h>
#include <hypar.h>

/*! 3D Navier-Stokes-specific function called at the beginning of each time-step:
    Solve and update the ceoncentration of reacting species and UV photons
*/
int NavierStokes3DPreStep( double  *u,   /*!< Solution (conserved variables) */
                           void    *s,   /*!< Solver object of type #HyPar */
                           void    *m,   /*!< MPI object of type #MPIVariables */
                           double  dt,   /*!< Time step size */
                           double  waqt  /*!< Current solution time */ )
{
  HyPar* solver = (HyPar*) s;
  NavierStokes3D* param = (NavierStokes3D*) solver->physics;

  return 0;
}
