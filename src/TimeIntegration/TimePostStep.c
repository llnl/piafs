// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file TimePostStep.c
    @brief Post-time-step function
    @author Debojyoti Ghosh
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <mpivars.h>
#include <simulation_object.h>
#include <timeintegration.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#endif

/*!
  Post-time-step function: this function is called at the end of
  each time step.
  + It updates the current simulation time.
  + It calls functions to print information and to write
    transient solution to file.
  + It will also call any physics-specific post-time-step function,
    if defined.
*/
int TimePostStep(void *ts /*!< Object of type #TimeIntegration */)
{
  TimeIntegration* TS = (TimeIntegration*) ts;
  SimulationObject* sim = (SimulationObject*) TS->simulation;
  int ns, nsims = TS->nsims;

  /* update current time */
  TS->waqt += TS->dt;

  if ((TS->iter+1)%sim[0].solver.screen_op_iter == 0) {

    /* Calculate norm for this time step */
    double sum = 0.0;
    double npts = 0;
    for (ns = 0; ns < nsims; ns++) {
#ifdef GPU_CUDA
      if (GPUShouldUse()) {
        /* Use GPU operations */
        GPUArrayAXPY(sim[ns].solver.u, -1.0, TS->u+TS->u_offsets[ns], TS->u_sizes[ns]);
        if (GPUShouldSyncEveryOp()) GPUSync();
        /* ArraySumSquarenD needs host array - copy to host first */
        int size = TS->u_sizes[ns];
        double *u_temp_host = (double*) malloc(size * sizeof(double));
        if (u_temp_host) {
          GPUCopyToHost(u_temp_host, TS->u+TS->u_offsets[ns], size * sizeof(double));
          sum += ArraySumSquarenD( sim[ns].solver.nvars,
                                   sim[ns].solver.ndims,
                                   sim[ns].solver.dim_local,
                                   sim[ns].solver.ghosts,
                                   sim[ns].solver.index,
                                   u_temp_host );
          free(u_temp_host);
        }
      } else {
        _ArrayAXPY_(sim[ns].solver.u,-1.0,(TS->u+TS->u_offsets[ns]),TS->u_sizes[ns]);
        sum += ArraySumSquarenD( sim[ns].solver.nvars,
                                 sim[ns].solver.ndims,
                                 sim[ns].solver.dim_local,
                                 sim[ns].solver.ghosts,
                                 sim[ns].solver.index,
                                 (TS->u+TS->u_offsets[ns]) );
      }
#elif defined(GPU_HIP)
      if (GPUShouldUse()) {
        /* Use GPU operations */
        GPUArrayAXPY(sim[ns].solver.u, -1.0, TS->u+TS->u_offsets[ns], TS->u_sizes[ns]);
        if (GPUShouldSyncEveryOp()) GPUSync();
        /* ArraySumSquarenD needs host array - copy to host first */
        int size = TS->u_sizes[ns];
        double *u_temp_host = (double*) malloc(size * sizeof(double));
        if (u_temp_host) {
          GPUCopyToHost(u_temp_host, TS->u+TS->u_offsets[ns], size * sizeof(double));
          sum += ArraySumSquarenD( sim[ns].solver.nvars,
                                   sim[ns].solver.ndims,
                                   sim[ns].solver.dim_local,
                                   sim[ns].solver.ghosts,
                                   sim[ns].solver.index,
                                   u_temp_host );
          free(u_temp_host);
        }
      } else {
        _ArrayAXPY_(sim[ns].solver.u,-1.0,(TS->u+TS->u_offsets[ns]),TS->u_sizes[ns]);
        sum += ArraySumSquarenD( sim[ns].solver.nvars,
                                 sim[ns].solver.ndims,
                                 sim[ns].solver.dim_local,
                                 sim[ns].solver.ghosts,
                                 sim[ns].solver.index,
                                 (TS->u+TS->u_offsets[ns]) );
      }
#else
      _ArrayAXPY_(sim[ns].solver.u,-1.0,(TS->u+TS->u_offsets[ns]),TS->u_sizes[ns]);
      sum += ArraySumSquarenD( sim[ns].solver.nvars,
                               sim[ns].solver.ndims,
                               sim[ns].solver.dim_local,
                               sim[ns].solver.ghosts,
                               sim[ns].solver.index,
                               (TS->u+TS->u_offsets[ns]) );
#endif
      npts += (double)sim[ns].solver.npoints_global;
    }

    double global_sum = 0;
    MPISum_double(  &global_sum,
                    &sum,1,
                    &(sim[0].mpi.world) );

    if (npts == 0) {
      fprintf(stderr,"ERROR in TimePostStep: Total grid points is zero, cannot compute norm.\n");
      exit(1);
    }
    TS->norm = sqrt(global_sum/npts);
    if (isnan(TS->norm) || isinf(TS->norm)) {
      fprintf(stderr,"ERROR in TimePostStep: NaN/Inf detected in residual norm at iteration %d.\n",TS->iter+1);
      exit(1);
    }

    /* write to file */
    if (TS->ResidualFile) {
      fprintf((FILE*)TS->ResidualFile,"%10d\t%E\t%E\n",TS->iter+1,TS->waqt,TS->norm);
    }

  }


  for (ns = 0; ns < nsims; ns++) {

    if (!strcmp(sim[ns].solver.ConservationCheck,"yes")) {
      /* calculate volume integral of the solution at this time step */
#ifdef GPU_CUDA
      if (GPUShouldUse()) {
        /* VolumeIntegral accesses u - need to copy to host. dxinv is already on host */
        int size_u = sim[ns].solver.npoints_local_wghosts * sim[ns].solver.nvars;
        double *u_host = (double*) malloc(size_u * sizeof(double));
        if (u_host) {
          GPUCopyToHost(u_host, sim[ns].solver.u, size_u * sizeof(double));
          IERR sim[ns].solver.VolumeIntegralFunction( sim[ns].solver.VolumeIntegral,
                                                      u_host,
                                                      &(sim[ns].solver),
                                                      &(sim[ns].mpi) ); CHECKERR(ierr);
        }
        if (u_host) free(u_host);
      } else {
        IERR sim[ns].solver.VolumeIntegralFunction( sim[ns].solver.VolumeIntegral,
                                                    sim[ns].solver.u,
                                                    &(sim[ns].solver),
                                                    &(sim[ns].mpi) ); CHECKERR(ierr);
      }
#elif defined(GPU_HIP)
      if (GPUShouldUse()) {
        /* VolumeIntegral accesses u - need to copy to host. dxinv is already on host */
        int size_u = sim[ns].solver.npoints_local_wghosts * sim[ns].solver.nvars;
        double *u_host = (double*) malloc(size_u * sizeof(double));
        if (u_host) {
          GPUCopyToHost(u_host, sim[ns].solver.u, size_u * sizeof(double));
          IERR sim[ns].solver.VolumeIntegralFunction( sim[ns].solver.VolumeIntegral,
                                                      u_host,
                                                      &(sim[ns].solver),
                                                      &(sim[ns].mpi) ); CHECKERR(ierr);
        }
        if (u_host) free(u_host);
      } else {
        IERR sim[ns].solver.VolumeIntegralFunction( sim[ns].solver.VolumeIntegral,
                                                    sim[ns].solver.u,
                                                    &(sim[ns].solver),
                                                    &(sim[ns].mpi) ); CHECKERR(ierr);
      }
#else
      IERR sim[ns].solver.VolumeIntegralFunction( sim[ns].solver.VolumeIntegral,
                                                  sim[ns].solver.u,
                                                  &(sim[ns].solver),
                                                  &(sim[ns].mpi) ); CHECKERR(ierr);
#endif
      /* calculate surface integral of the flux at this time step */
#ifdef GPU_CUDA
      if (GPUShouldUse()) {
        /* BoundaryIntegral accesses StepBoundaryIntegral - need host copy. dxinv is already on host */
        int bf_size = 2*sim[ns].solver.ndims*sim[ns].solver.nvars;
        double *StepBoundaryIntegral_host = (double*) malloc(bf_size * sizeof(double));

        if (StepBoundaryIntegral_host) {
          GPUCopyToHost(StepBoundaryIntegral_host, sim[ns].solver.StepBoundaryIntegral, bf_size * sizeof(double));

          /* Temporarily swap pointer */
          double *StepBoundaryIntegral_save = sim[ns].solver.StepBoundaryIntegral;
          sim[ns].solver.StepBoundaryIntegral = StepBoundaryIntegral_host;

          IERR sim[ns].solver.BoundaryIntegralFunction( &(sim[ns].solver),
                                                        &(sim[ns].mpi)); CHECKERR(ierr);

          /* Restore original pointer */
          sim[ns].solver.StepBoundaryIntegral = StepBoundaryIntegral_save;

          free(StepBoundaryIntegral_host);
        } else {
          fprintf(stderr, "Error: Failed to allocate host buffer for BoundaryIntegral\n");
          return 1;
        }
      } else {
        IERR sim[ns].solver.BoundaryIntegralFunction( &(sim[ns].solver),
                                                      &(sim[ns].mpi)); CHECKERR(ierr);
      }
#elif defined(GPU_HIP)
      if (GPUShouldUse()) {
        /* BoundaryIntegral accesses StepBoundaryIntegral - need host copy. dxinv is already on host */
        int bf_size = 2*sim[ns].solver.ndims*sim[ns].solver.nvars;
        double *StepBoundaryIntegral_host = (double*) malloc(bf_size * sizeof(double));

        if (StepBoundaryIntegral_host) {
          GPUCopyToHost(StepBoundaryIntegral_host, sim[ns].solver.StepBoundaryIntegral, bf_size * sizeof(double));

          /* Temporarily swap pointer */
          double *StepBoundaryIntegral_save = sim[ns].solver.StepBoundaryIntegral;
          sim[ns].solver.StepBoundaryIntegral = StepBoundaryIntegral_host;

          IERR sim[ns].solver.BoundaryIntegralFunction( &(sim[ns].solver),
                                                        &(sim[ns].mpi)); CHECKERR(ierr);

          /* Restore original pointer */
          sim[ns].solver.StepBoundaryIntegral = StepBoundaryIntegral_save;

          free(StepBoundaryIntegral_host);
        } else {
          fprintf(stderr, "Error: Failed to allocate host buffer for BoundaryIntegral\n");
          return 1;
        }
      } else {
        IERR sim[ns].solver.BoundaryIntegralFunction( &(sim[ns].solver),
                                                      &(sim[ns].mpi)); CHECKERR(ierr);
      }
#else
      IERR sim[ns].solver.BoundaryIntegralFunction( &(sim[ns].solver),
                                                    &(sim[ns].mpi)); CHECKERR(ierr);
#endif
      /* calculate the conservation error at this time step       */
      IERR sim[ns].solver.CalculateConservationError( &(sim[ns].solver),
                                                      &(sim[ns].mpi)); CHECKERR(ierr);
    }

    if (sim[ns].solver.PostStep) {
      /* PostStep receives u which is on GPU - if it needs to access it, it should be GPU-aware */
      /* For now, assume PostStep doesn't access arrays directly or is GPU-aware */
      sim[ns].solver.PostStep(sim[ns].solver.u,&(sim[ns].solver),&(sim[ns].mpi),TS->waqt,TS->iter);
    }

  }

  gettimeofday(&TS->iter_end_time,NULL);
  long long walltime;
  walltime = (  (TS->iter_end_time.tv_sec * 1000000 + TS->iter_end_time.tv_usec)
              - (TS->iter_start_time.tv_sec * 1000000 + TS->iter_start_time.tv_usec));
  TS->iter_wctime = (double) walltime / 1000000.0;
  TS->iter_wctime_total += TS->iter_wctime;

  double global_total = 0, global_wctime = 0, global_mpi_total = 0, global_mpi_wctime = 0;

  MPIMax_double(&global_wctime, &TS->iter_wctime, 1, &(sim[0].mpi.world));
  MPIMax_double(&global_total, &TS->iter_wctime_total, 1, &(sim[0].mpi.world));

  return(0);
}

