// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file TimePreStep.c
    @brief Pre-time-step function
    @author Debojyoti Ghosh
*/

#include <basic.h>
#include <arrayfunctions.h>
#include <timeintegration.h>
#include <mpivars.h>
#include <simulation_object.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#include <gpu_mpi.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#include <gpu_mpi.h>
#endif

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
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      solver->ApplyBoundaryConditions( solver,
                                       mpi,
                                       u,
                                       NULL,
                                       TS->waqt );

      /* Use GPU-aware MPI exchange */
      GPUMPIExchangeBoundariesnD( solver->ndims,
                                  solver->nvars,
                                  solver->dim_local,
                                  solver->ghosts,
                                  mpi,
                                  u );
    } else {
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
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      solver->ApplyBoundaryConditions( solver,
                                       mpi,
                                       u,
                                       NULL,
                                       TS->waqt );

      /* Use GPU-aware MPI exchange */
      GPUMPIExchangeBoundariesnD( solver->ndims,
                                  solver->nvars,
                                  solver->dim_local,
                                  solver->ghosts,
                                  mpi,
                                  u );
    } else {
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
    }
#else
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
#endif

    if ((TS->iter+1)%solver->screen_op_iter == 0) {

#ifdef GPU_CUDA
      if (GPUShouldUse()) {
        /* Copy from GPU to host for CFL computation */
        GPUArrayCopy(TS->u + TS->u_offsets[ns], solver->u, solver->npoints_local_wghosts*solver->nvars);
        if (GPUShouldSyncEveryOp()) GPUSync();
      } else {
        _ArrayCopy1D_(  solver->u,
                        (TS->u + TS->u_offsets[ns]),
                        (solver->npoints_local_wghosts*solver->nvars) );
      }
#elif defined(GPU_HIP)
      if (GPUShouldUse()) {
        /* Copy from GPU to host for CFL computation */
        GPUArrayCopy(TS->u + TS->u_offsets[ns], solver->u, solver->npoints_local_wghosts*solver->nvars);
        if (GPUShouldSyncEveryOp()) GPUSync();
      } else {
        _ArrayCopy1D_(  solver->u,
                        (TS->u + TS->u_offsets[ns]),
                        (solver->npoints_local_wghosts*solver->nvars) );
      }
#else
      _ArrayCopy1D_(  solver->u,
                      (TS->u + TS->u_offsets[ns]),
                      (solver->npoints_local_wghosts*solver->nvars) );
#endif

      /* compute max CFL over the domain */
      if (solver->ComputeCFL) {
        /* ComputeCFL is now GPU-aware, can be called directly */
        double local_max_cfl = solver->ComputeCFL(solver, mpi, TS->dt, TS->waqt);
        MPIMax_double(&TS->max_cfl ,&local_max_cfl ,1,&mpi->world);
      } else {
        TS->max_cfl = -1;
      }

    }

    /* set the step boundary flux integral value to zero */
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      GPUArraySetValue(solver->StepBoundaryIntegral, 0.0, 2*solver->ndims*solver->nvars);
    } else {
      _ArraySetValue_(solver->StepBoundaryIntegral,2*solver->ndims*solver->nvars,0.0);
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      GPUArraySetValue(solver->StepBoundaryIntegral, 0.0, 2*solver->ndims*solver->nvars);
    } else {
      _ArraySetValue_(solver->StepBoundaryIntegral,2*solver->ndims*solver->nvars,0.0);
    }
#else
    _ArraySetValue_(solver->StepBoundaryIntegral,2*solver->ndims*solver->nvars,0.0);
#endif

    if (solver->PreStep) {
      /* PreStep receives u which is on GPU - if it needs to access it, it should be GPU-aware */
      /* For now, NavierStokes3DPreStep does nothing, so this is safe */
      solver->PreStep(u,solver,mpi,TS->dt,TS->waqt);
    }

  }

  return 0;
}
