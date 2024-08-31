/*! @file TimePreStep.c
    @brief Pre-time-step function
    @author Debojyoti Ghosh
*/

#include <basic.h>
#include <arrayfunctions.h>
#include <timeintegration.h>
#include <mpivars.h>
#include <simulation_object.h>

/*!
  Pre-time-step function: This function is called before each time
  step. Some notable things this does are:
  + Computes CFL.
  + Call the physics-specific pre-time-step function, if defined.
*/
int TimePreStep(void *ts /*!< Object of type #TimeIntegration */ )
{
  TimeIntegration*  TS  = (TimeIntegration*) ts;
  _DECLARE_IERR_;

  SimulationObject* sim = (SimulationObject*) TS->simulation;
  int ns, nsims = TS->nsims;

  gettimeofday(&TS->iter_start_time,NULL);

  for (ns = 0; ns < nsims; ns++) {

    HyPar*        solver = &(sim[ns].solver);
    MPIVariables* mpi    = &(sim[ns].mpi);

    double *u = NULL;
    u = solver->u;

    /* apply boundary conditions and exchange data over MPI interfaces */

    solver->ApplyBoundaryConditions( solver,
                                     mpi,
                                     u,
                                     NULL,
                                     TS->waqt );

    MPIExchangeBoundariesnD( solver->ndims,
                             solver->nvars,
                             solver->dim_local,
                             solver->ghosts,
                             mpi,
                             u );

    if ((TS->iter+1)%solver->screen_op_iter == 0) {

      _ArrayCopy1D_(  solver->u,
                      (TS->u + TS->u_offsets[ns]),
                      (solver->npoints_local_wghosts*solver->nvars) );

      /* compute max CFL over the domain */
      if (solver->ComputeCFL) {
        double local_max_cfl  = -1.0;
        local_max_cfl  = solver->ComputeCFL (solver,mpi,TS->dt,TS->waqt);
        MPIMax_double(&TS->max_cfl ,&local_max_cfl ,1,&mpi->world);
      } else {
        TS->max_cfl = -1;
      }

    }

    /* set the step boundary flux integral value to zero */
    _ArraySetValue_(solver->StepBoundaryIntegral,2*solver->ndims*solver->nvars,0.0);

    if (solver->PreStep) {
      solver->PreStep(u,solver,mpi,TS->dt,TS->waqt);
    }

  }

  return 0;
}
