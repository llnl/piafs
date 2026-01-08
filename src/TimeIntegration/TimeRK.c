/*! @file TimeRK.c
    @brief Explicit Runge-Kutta method
    @author Debojyoti Ghosh
*/

#include <stdio.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <simulation_object.h>
#include <timeintegration.h>
#include <time.h>
#include <math.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#include <gpu_kernels.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#include <gpu_kernels.h>
#endif

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
  
  if (!params) {
    return 1;
  }
  
  
  int ns, stage, i, nsims = TS->nsims;

  /* Calculate stage values */
  for (stage = 0; stage < params->nstages; stage++) {

    double stagetime = TS->waqt + params->c[stage]*TS->dt;


#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      /* Copy solver->u to TS->U[stage] on GPU */
      for (ns = 0; ns < nsims; ns++) {
        GPUArrayCopy(TS->U[stage] + TS->u_offsets[ns],
                     sim[ns].solver.u,
                     TS->u_sizes[ns]);
      }

      /* Compute stage values on GPU - use fused kernels for performance */
      if (stage == 0) {
        /* Stage 0: No computation needed, U[0] already copied */
      } else if (stage == 1) {
        /* Stage 1: Single AXPY */
        GPUArrayAXPY(TS->Udot[0],
                     TS->dt * params->A[stage*params->nstages+0],
                     TS->U[stage],
                     TS->u_size_total);
      } else if (stage == 2) {
        /* Stage 2: Fused 2-way AXPY */
        gpu_launch_array_axpy_chain2(
          TS->Udot[0], TS->dt * params->A[stage*params->nstages+0],
          TS->Udot[1], TS->dt * params->A[stage*params->nstages+1],
          TS->U[stage], TS->u_size_total, 512);
      } else if (stage == 3) {
        /* Stage 3: Fused 3-way AXPY */
        gpu_launch_array_axpy_chain3(
          TS->Udot[0], TS->dt * params->A[stage*params->nstages+0],
          TS->Udot[1], TS->dt * params->A[stage*params->nstages+1],
          TS->Udot[2], TS->dt * params->A[stage*params->nstages+2],
          TS->U[stage], TS->u_size_total, 512);
      } else if (stage == 4) {
        /* Stage 4: Fused 4-way AXPY */
        gpu_launch_array_axpy_chain4(
          TS->Udot[0], TS->dt * params->A[stage*params->nstages+0],
          TS->Udot[1], TS->dt * params->A[stage*params->nstages+1],
          TS->Udot[2], TS->dt * params->A[stage*params->nstages+2],
          TS->Udot[3], TS->dt * params->A[stage*params->nstages+3],
          TS->U[stage], TS->u_size_total, 512);
      } else {
        /* General case: Fallback to loop for higher-order methods */
        for (i = 0; i < stage; i++) {
          GPUArrayAXPY(TS->Udot[i],
                       TS->dt * params->A[stage*params->nstages+i],
                       TS->U[stage],
                       TS->u_size_total);
        }
      }
    } else {
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
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      /* Copy solver->u to TS->U[stage] on GPU */
      for (ns = 0; ns < nsims; ns++) {
        GPUArrayCopy(TS->U[stage] + TS->u_offsets[ns],
                     sim[ns].solver.u,
                     TS->u_sizes[ns]);
      }

      /* Compute stage values on GPU - use fused kernels for performance */
      if (stage == 0) {
        /* Stage 0: No computation needed, U[0] already copied */
      } else if (stage == 1) {
        /* Stage 1: Single AXPY */
        GPUArrayAXPY(TS->Udot[0],
                     TS->dt * params->A[stage*params->nstages+0],
                     TS->U[stage],
                     TS->u_size_total);
      } else if (stage == 2) {
        /* Stage 2: Fused 2-way AXPY */
        gpu_launch_array_axpy_chain2(
          TS->Udot[0], TS->dt * params->A[stage*params->nstages+0],
          TS->Udot[1], TS->dt * params->A[stage*params->nstages+1],
          TS->U[stage], TS->u_size_total, 512);
      } else if (stage == 3) {
        /* Stage 3: Fused 3-way AXPY */
        gpu_launch_array_axpy_chain3(
          TS->Udot[0], TS->dt * params->A[stage*params->nstages+0],
          TS->Udot[1], TS->dt * params->A[stage*params->nstages+1],
          TS->Udot[2], TS->dt * params->A[stage*params->nstages+2],
          TS->U[stage], TS->u_size_total, 512);
      } else if (stage == 4) {
        /* Stage 4: Fused 4-way AXPY */
        gpu_launch_array_axpy_chain4(
          TS->Udot[0], TS->dt * params->A[stage*params->nstages+0],
          TS->Udot[1], TS->dt * params->A[stage*params->nstages+1],
          TS->Udot[2], TS->dt * params->A[stage*params->nstages+2],
          TS->Udot[3], TS->dt * params->A[stage*params->nstages+3],
          TS->U[stage], TS->u_size_total, 512);
      } else {
        /* General case: Fallback to loop for higher-order methods */
        for (i = 0; i < stage; i++) {
          GPUArrayAXPY(TS->Udot[i],
                       TS->dt * params->A[stage*params->nstages+i],
                       TS->U[stage],
                       TS->u_size_total);
        }
      }
    } else {
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
    }
#else
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
#endif

    for (ns = 0; ns < nsims; ns++) {
      if (sim[ns].solver.PostStage) {
#ifdef GPU_CUDA
        if (GPUShouldUse()) {
          /* PostStage receives GPU array - if it needs to access it, copy to host first */
          /* For now, assume PostStage doesn't access arrays directly or is GPU-aware */
          sim[ns].solver.PostStage(  (TS->U[stage] + TS->u_offsets[ns]),
                                     &(sim[ns].solver),
                                     &(sim[ns].mpi),
                                     stagetime); CHECKERR(ierr);
        } else {
          sim[ns].solver.PostStage(  (TS->U[stage] + TS->u_offsets[ns]),
                                     &(sim[ns].solver),
                                     &(sim[ns].mpi),
                                     stagetime); CHECKERR(ierr);
        }
#elif defined(GPU_HIP)
        if (GPUShouldUse()) {
          /* PostStage receives GPU array - if it needs to access it, copy to host first */
          /* For now, assume PostStage doesn't access arrays directly or is GPU-aware */
          sim[ns].solver.PostStage(  (TS->U[stage] + TS->u_offsets[ns]),
                                     &(sim[ns].solver),
                                     &(sim[ns].mpi),
                                     stagetime); CHECKERR(ierr);
        } else {
          sim[ns].solver.PostStage(  (TS->U[stage] + TS->u_offsets[ns]),
                                     &(sim[ns].solver),
                                     &(sim[ns].mpi),
                                     stagetime); CHECKERR(ierr);
        }
#else
        sim[ns].solver.PostStage(  (TS->U[stage] + TS->u_offsets[ns]),
                                   &(sim[ns].solver),
                                   &(sim[ns].mpi),
                                   stagetime); CHECKERR(ierr);
#endif
      }
    }

    for (ns = 0; ns < nsims; ns++) {
      int rhs_ierr = TS->RHSFunction( (TS->Udot[stage] + TS->u_offsets[ns]),
                                      (TS->U[stage] + TS->u_offsets[ns]),
                                      &(sim[ns].solver),
                                      &(sim[ns].mpi),
                                      stagetime);
      if (rhs_ierr) {
        fprintf(stderr, "Error: TimeRK: RHSFunction failed (stage=%d, domain=%d, ierr=%d)\n",
                stage, ns, rhs_ierr);
        exit(1);
      }
    }

#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      /* BoundaryFlux is on host, so use CPU operations */
      _ArraySetValue_(TS->BoundaryFlux[stage], TS->bf_size_total, 0.0);
      for (ns = 0; ns < nsims; ns++) {
        /* StageBoundaryIntegral is on GPU, need to copy to host */
        double *bf_host = (double*) malloc(TS->bf_sizes[ns] * sizeof(double));
        GPUCopyToHost(bf_host, sim[ns].solver.StageBoundaryIntegral, TS->bf_sizes[ns] * sizeof(double));
        _ArrayCopy1D_(bf_host, (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]), TS->bf_sizes[ns]);
        free(bf_host);
      }
    } else {
      _ArraySetValue_(TS->BoundaryFlux[stage], TS->bf_size_total, 0.0);
      for (ns = 0; ns < nsims; ns++) {
        _ArrayCopy1D_(  sim[ns].solver.StageBoundaryIntegral,
                        (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]),
                        TS->bf_sizes[ns] );
      }
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      /* BoundaryFlux is on host, so use CPU operations */
      _ArraySetValue_(TS->BoundaryFlux[stage], TS->bf_size_total, 0.0);
      for (ns = 0; ns < nsims; ns++) {
        /* StageBoundaryIntegral is on GPU, need to copy to host */
        double *bf_host = (double*) malloc(TS->bf_sizes[ns] * sizeof(double));
        GPUCopyToHost(bf_host, sim[ns].solver.StageBoundaryIntegral, TS->bf_sizes[ns] * sizeof(double));
        _ArrayCopy1D_(bf_host, (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]), TS->bf_sizes[ns]);
        free(bf_host);
      }
    } else {
      _ArraySetValue_(TS->BoundaryFlux[stage], TS->bf_size_total, 0.0);
      for (ns = 0; ns < nsims; ns++) {
        _ArrayCopy1D_(  sim[ns].solver.StageBoundaryIntegral,
                        (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]),
                        TS->bf_sizes[ns] );
      }
    }
#else
    _ArraySetValue_(TS->BoundaryFlux[stage], TS->bf_size_total, 0.0);
    for (ns = 0; ns < nsims; ns++) {
      _ArrayCopy1D_(  sim[ns].solver.StageBoundaryIntegral,
                      (TS->BoundaryFlux[stage] + TS->bf_offsets[ns]),
                      TS->bf_sizes[ns] );
    }
#endif

  }

  /* Step completion */
  for (stage = 0; stage < params->nstages; stage++) {

#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      for (ns = 0; ns < nsims; ns++) {
        GPUArrayAXPY(TS->Udot[stage] + TS->u_offsets[ns],
                     TS->dt * params->b[stage],
                     sim[ns].solver.u,
                     TS->u_sizes[ns]);
        /* StepBoundaryIntegral is on GPU, BoundaryFlux is on host */
        double *bf_host = (double*) malloc(TS->bf_sizes[ns] * sizeof(double));
        _ArrayCopy1D_((TS->BoundaryFlux[stage] + TS->bf_offsets[ns]), bf_host, TS->bf_sizes[ns]);
        double *sbi_host = (double*) malloc(TS->bf_sizes[ns] * sizeof(double));
        GPUCopyToHost(sbi_host, sim[ns].solver.StepBoundaryIntegral, TS->bf_sizes[ns] * sizeof(double));
        for (int j = 0; j < TS->bf_sizes[ns]; j++) {
          sbi_host[j] += TS->dt * params->b[stage] * bf_host[j];
        }
        GPUCopyToDevice(sim[ns].solver.StepBoundaryIntegral, sbi_host, TS->bf_sizes[ns] * sizeof(double));
        free(bf_host);
        free(sbi_host);
      }
    } else {
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
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      for (ns = 0; ns < nsims; ns++) {
        GPUArrayAXPY(TS->Udot[stage] + TS->u_offsets[ns],
                     TS->dt * params->b[stage],
                     sim[ns].solver.u,
                     TS->u_sizes[ns]);
        /* StepBoundaryIntegral is on GPU, BoundaryFlux is on host */
        double *bf_host = (double*) malloc(TS->bf_sizes[ns] * sizeof(double));
        _ArrayCopy1D_((TS->BoundaryFlux[stage] + TS->bf_offsets[ns]), bf_host, TS->bf_sizes[ns]);
        double *sbi_host = (double*) malloc(TS->bf_sizes[ns] * sizeof(double));
        GPUCopyToHost(sbi_host, sim[ns].solver.StepBoundaryIntegral, TS->bf_sizes[ns] * sizeof(double));
        for (int j = 0; j < TS->bf_sizes[ns]; j++) {
          sbi_host[j] += TS->dt * params->b[stage] * bf_host[j];
        }
        GPUCopyToDevice(sim[ns].solver.StepBoundaryIntegral, sbi_host, TS->bf_sizes[ns] * sizeof(double));
        free(bf_host);
        free(sbi_host);
      }
    } else {
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
#else
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
#endif

  }

  /* Check for NaN/Inf in final solution after all stages
     Note: This is expensive on GPU (requires cudaMemcpy). Disabled by default
     for Release builds. Enable with -DENABLE_NAN_CHECK=ON or Debug build. */
#ifdef PIAFS_NAN_CHECK
#ifdef GPU_CUDA
  if (GPUShouldUse()) {
    /* Copy to host for NaN check */
    for (ns = 0; ns < nsims; ns++) {
      double *u_host = (double*) malloc(TS->u_sizes[ns] * sizeof(double));
      GPUCopyToHost(u_host, sim[ns].solver.u, TS->u_sizes[ns] * sizeof(double));
      for (int i = 0; i < TS->u_sizes[ns]; i++) {
        if (isnan(u_host[i]) || isinf(u_host[i])) {
          fprintf(stderr,"ERROR in TimeRK: NaN/Inf detected in solution at index %d, time=%e, dt=%e.\n",i,TS->waqt,TS->dt);
          free(u_host);
          exit(1);
        }
      }
      free(u_host);
    }
  } else {
    for (ns = 0; ns < nsims; ns++) {
      for (int i = 0; i < TS->u_sizes[ns]; i++) {
        if (isnan(sim[ns].solver.u[i]) || isinf(sim[ns].solver.u[i])) {
          fprintf(stderr,"ERROR in TimeRK: NaN/Inf detected in solution at index %d, time=%e, dt=%e.\n",i,TS->waqt,TS->dt);
          exit(1);
        }
      }
    }
  }
#elif defined(GPU_HIP)
  if (GPUShouldUse()) {
    /* Copy to host for NaN check */
    for (ns = 0; ns < nsims; ns++) {
      double *u_host = (double*) malloc(TS->u_sizes[ns] * sizeof(double));
      GPUCopyToHost(u_host, sim[ns].solver.u, TS->u_sizes[ns] * sizeof(double));
      for (int i = 0; i < TS->u_sizes[ns]; i++) {
        if (isnan(u_host[i]) || isinf(u_host[i])) {
          fprintf(stderr,"ERROR in TimeRK: NaN/Inf detected in solution at index %d, time=%e, dt=%e.\n",i,TS->waqt,TS->dt);
          free(u_host);
          exit(1);
        }
      }
      free(u_host);
    }
  } else {
    for (ns = 0; ns < nsims; ns++) {
      for (int i = 0; i < TS->u_sizes[ns]; i++) {
        if (isnan(sim[ns].solver.u[i]) || isinf(sim[ns].solver.u[i])) {
          fprintf(stderr,"ERROR in TimeRK: NaN/Inf detected in solution at index %d, time=%e, dt=%e.\n",i,TS->waqt,TS->dt);
          exit(1);
        }
      }
    }
  }
#else
  for (ns = 0; ns < nsims; ns++) {
    for (int i = 0; i < TS->u_sizes[ns]; i++) {
      if (isnan(sim[ns].solver.u[i]) || isinf(sim[ns].solver.u[i])) {
        fprintf(stderr,"ERROR in TimeRK: NaN/Inf detected in solution at index %d, time=%e, dt=%e.\n",i,TS->waqt,TS->dt);
        exit(1);
      }
    }
  }
#endif
#endif /* PIAFS_NAN_CHECK */

  return 0;
}

