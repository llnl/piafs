/*! @file Euler1DPreStep.c
    @author Debojyoti Ghosh
    @brief Contains the 1D Euler-specific function to be called at the beginning of each time step.
*/

#include <physicalmodels/euler1d.h>
#include <hypar.h>

/*! 1D Euler-specific function called at the beginning of each time-step:
    Solve and update the ceoncentration of reacting species and UV photons
*/
int Euler1DPreStep( double  *u,   /*!< Solution (conserved variables) */
                    void    *s,   /*!< Solver object of type #HyPar */
                    void    *m,   /*!< MPI object of type #MPIVariables */
                    double  dt,   /*!< Time step size */
                    double  waqt  /*!< Current solution time */ )
{
  HyPar   *solver = (HyPar*)   s;
  Euler1D *param  = (Euler1D*) solver->physics;

  return 0;
}
