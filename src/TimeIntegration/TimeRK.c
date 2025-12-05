/*! @file TimeRK.c
    @brief Explicit Runge-Kutta method
    @author Debojyoti Ghosh
*/

#include <basic.h>
#include <arrayfunctions.h>
#include <simulation_object.h>
#include <timeintegration.h>
#include <time.h>
#include <math.h>

/*!
  Advance the ODE given by
  \f{equation}{
    \frac{d{\bf u}}{dt} = {\bf F} \left({\bf u}\right)
  \f}
  by one time step of size #HyPar::dt using the forward Euler method
  given by
  \f{align}{
    {\bf U}^{\left(i\right)} &= {\bf u}_n + \Delta t \sum_{j=1}^{i-1} a_{ij} {\bf F}\left({\bf U}^{\left(j\right)}\right), \\
    {\bf u}_{n+1} &= {\bf u}_n + \Delta t \sum_{i=1}^s b_{i} {\bf F}\left({\bf U}^{\left(i\right)}\right),
  \f}
  where the subscript represents the time level, the superscripts represent the stages, \f$\Delta t\f$ is the
  time step size #HyPar::dt, and \f${\bf F}\left({\bf u}\right)\f$ is computed by #TimeIntegration::RHSFunction.
  The Butcher tableaux coefficients are \f$a_{ij}\f$ (#ExplicitRKParameters::A) and \f$b_i\f$
  (#ExplicitRKParameters::b).

  Note: In the code #TimeIntegration::Udot is equivalent to \f${\bf F}\left({\bf u}\right)\f$.
*/
int TimeRK(void *ts /*!< Object of type #TimeIntegration */)
{
  TimeIntegration* TS = (TimeIntegration*) ts;
  SimulationObject* sim = (SimulationObject*) TS->simulation;
  ExplicitRKParameters *params = (ExplicitRKParameters*) sim[0].solver.msti;
  int ns, stage, i, nsims = TS->nsims;

  /* Calculate stage values */
  for (stage = 0; stage < params->nstages; stage++) {

    double stagetime = TS->waqt + params->c[stage]*TS->dt;

    for (ns = 0; ns < nsims; ns++) {
      _ArrayCopy1D_(  sim[ns].solver.u,
                      (TS->U[stage] + TS->u_offsets[ns]),
                      (TS->u_sizes[ns]) );
    }

    for (i = 0; i < stage; i++) {
      _ArrayAXPY_(  TS->Udot[i],
                    (TS->dt * params->A[stage*params->nstages+i]),
                    TS->U[stage],
                    TS->u_size_total );
    }

    for (ns = 0; ns < nsims; ns++) {
      if (sim[ns].solver.PostStage) {
        sim[ns].solver.PostStage(  (TS->U[stage] + TS->u_offsets[ns]),
                                   &(sim[ns].solver),
                                   &(sim[ns].mpi),
                                   stagetime); CHECKERR(ierr);
      }
    }

    for (ns = 0; ns < nsims; ns++) {
      TS->RHSFunction( (TS->Udot[stage] + TS->u_offsets[ns]),
                       (TS->U[stage] + TS->u_offsets[ns]),
                       &(sim[ns].solver),
                       &(sim[ns].mpi),
                       stagetime);
    }

    _ArraySetValue_(TS->BoundaryFlux[stage], TS->bf_size_total, 0.0);
    for (ns = 0; ns < nsims; ns++) {
      _ArrayCopy1D_(  sim[ns].solver.StageBoundaryIntegral,
                      (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]),
                      TS->bf_sizes[ns] );
    }

  }

  /* Step completion */
  for (stage = 0; stage < params->nstages; stage++) {

    for (ns = 0; ns < nsims; ns++) {
      _ArrayAXPY_(  (TS->Udot[stage] + TS->u_offsets[ns]),
                    (TS->dt * params->b[stage]),
                    (sim[ns].solver.u),
                    (TS->u_sizes[ns]) );
      _ArrayAXPY_(  (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]),
                    (TS->dt * params->b[stage]),
                    (sim[ns].solver.StepBoundaryIntegral),
                    (TS->bf_sizes[ns]) );
    }

  }

  /* Check for NaN/Inf in final solution after all stages */
  for (ns = 0; ns < nsims; ns++) {
    for (int i = 0; i < TS->u_sizes[ns]; i++) {
      if (isnan(sim[ns].solver.u[i]) || isinf(sim[ns].solver.u[i])) {
        fprintf(stderr,"ERROR in TimeRK: NaN/Inf detected in solution at index %d, time=%e, dt=%e.\n",i,TS->waqt,TS->dt);
        exit(1);
      }
    }
  }

  return 0;
}

