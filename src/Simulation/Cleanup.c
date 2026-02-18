// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file Cleanup.c
    @author Debojyoti Ghosh
    @brief Clean up and free memory after simulation is complete.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <tridiagLU.h>
#include <boundaryconditions.h>
#include <timeintegration.h>
#include <interpolation.h>
#include <mpivars.h>
#include <simulation_object.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#endif

/* include header files for each physical model */
#include <physicalmodels/euler1d.h>
#include <physicalmodels/navierstokes2d.h>
#include <physicalmodels/navierstokes3d.h>

/*! Cleans up and frees the memory after the completion of the simulation. */
int Cleanup(  void  *s,   /*!< Array of simulation objects of type #SimulationObject */
              int   nsims /*!< number of simulation objects */
           )
{
  SimulationObject* sim = (SimulationObject*) s;
  int ns;
  _DECLARE_IERR_;

  if (nsims == 0) return 0;

  if (!sim[0].mpi.rank) {
    printf("Deallocating arrays.\n");
  }

  for (ns = 0; ns < nsims; ns++) {

    if (sim[ns].is_barebones == 1) {
      fprintf(stderr, "Error in Cleanup(): object is barebones type.\n");
      return 1;
    }

    HyPar* solver = &(sim[ns].solver);
    MPIVariables* mpi = &(sim[ns].mpi);
    DomainBoundary* boundary = (DomainBoundary*) solver->boundary;
    int i;

    /* Clean up boundary zones */
    for (i = 0; i < solver->nBoundaryZones; i++) {
      BCCleanup(&boundary[i]);
    }
    free(solver->boundary);

    /* Clean up any allocations in physical model */
    if (!strcmp(solver->model,_EULER_1D_)) {
      IERR Euler1DCleanup(solver->physics); CHECKERR(ierr);
    } else if (!strcmp(solver->model,_NAVIER_STOKES_2D_)) {
      IERR NavierStokes2DCleanup(solver->physics); CHECKERR(ierr);
    } else if (!strcmp(solver->model,_NAVIER_STOKES_3D_)) {
      IERR NavierStokes3DCleanup(solver->physics); CHECKERR(ierr);
    }
    free(solver->physics);

    /* Clean up any allocations from time-integration */
    if (!strcmp(solver->time_scheme,_RK_)) {
      IERR TimeExplicitRKCleanup(solver->msti); CHECKERR(ierr);
      free(solver->msti);
    }

    /* Clean up any spatial reconstruction related allocations */
    if (   (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_WENO_  ))
        || (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_CRWENO_)) ) {
      WENOCleanup(solver->interp);
    }
    if (solver->interp)   free(solver->interp);
    if (   (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_COMPACT_UPWIND_ ))
        || (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_CRWENO_         )) ) {
      IERR CompactSchemeCleanup(solver->compact); CHECKERR(ierr);
    }
    if (solver->compact)  free(solver->compact);
    if (solver->lusolver) free(solver->lusolver);

    /* Free the communicators created */
    IERR MPIFreeCommunicators(solver->ndims,mpi); CHECKERR(ierr);

    /* These variables are allocated in Initialize.c */
    free(solver->dim_global);
    free(solver->dim_global_ex);
    free(solver->dim_local);
    free(solver->index);
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      GPUFree(solver->u);
      GPUFree(solver->x);
      GPUFree(solver->dxinv);
      GPUFree(solver->hyp);
      GPUFree(solver->par);
      GPUFree(solver->source);
      GPUFree(solver->fluxI);
      GPUFree(solver->uL);
      GPUFree(solver->uR);
      GPUFree(solver->fL);
      GPUFree(solver->fR);
      GPUFree(solver->StageBoundaryIntegral);
      GPUFree(solver->StepBoundaryIntegral);
      /* Allocated on GPU in GPUAllocateSolutionArrays */
      GPUFree(solver->uC);
      GPUFree(solver->fluxC);
      GPUFree(solver->Deriv1);
      GPUFree(solver->Deriv2);
    } else {
      free(solver->u);
      free(solver->x);
      free(solver->dxinv);
      free(solver->hyp);
      free(solver->par);
      free(solver->source);
      free(solver->fluxI);
      free(solver->uL);
      free(solver->uR);
      free(solver->fL);
      free(solver->fR);
      free(solver->StageBoundaryIntegral);
      free(solver->StepBoundaryIntegral);
      free(solver->uC);
      free(solver->fluxC);
      free(solver->Deriv1);
      free(solver->Deriv2);
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      GPUFree(solver->u);
      GPUFree(solver->x);
      GPUFree(solver->dxinv);
      GPUFree(solver->hyp);
      GPUFree(solver->par);
      GPUFree(solver->source);
      GPUFree(solver->fluxI);
      GPUFree(solver->uL);
      GPUFree(solver->uR);
      GPUFree(solver->fL);
      GPUFree(solver->fR);
      GPUFree(solver->StageBoundaryIntegral);
      GPUFree(solver->StepBoundaryIntegral);
      /* Allocated on GPU in GPUAllocateSolutionArrays */
      GPUFree(solver->uC);
      GPUFree(solver->fluxC);
      GPUFree(solver->Deriv1);
      GPUFree(solver->Deriv2);
    } else {
      free(solver->u);
      free(solver->x);
      free(solver->dxinv);
      free(solver->hyp);
      free(solver->par);
      free(solver->source);
      free(solver->fluxI);
      free(solver->uL);
      free(solver->uR);
      free(solver->fL);
      free(solver->fR);
      free(solver->StageBoundaryIntegral);
      free(solver->StepBoundaryIntegral);
      free(solver->uC);
      free(solver->fluxC);
      free(solver->Deriv1);
      free(solver->Deriv2);
    }
#else
    free(solver->u);
    free(solver->x);
    free(solver->dxinv);
    free(solver->hyp);
    free(solver->par);
    free(solver->source);
    free(solver->fluxI);
    free(solver->uL);
    free(solver->uR);
    free(solver->fL);
    free(solver->fR);
    free(solver->StageBoundaryIntegral);
    free(solver->StepBoundaryIntegral);
    free(solver->uC);
    free(solver->fluxC);
    free(solver->Deriv1);
    free(solver->Deriv2);
#endif
    free(solver->isPeriodic);
    free(mpi->iproc);
    free(mpi->ip);
    free(mpi->is);
    free(mpi->ie);
    free(mpi->bcperiodic);
#if defined(GPU_CUDA) || defined(GPU_HIP)
    if (mpi->use_gpu_pinned) {
      GPUFreePinned(mpi->sendbuf);
      GPUFreePinned(mpi->recvbuf);
    } else
#endif
    {
      free(mpi->sendbuf);
      free(mpi->recvbuf);
    }
    free(solver->VolumeIntegral);
    free(solver->VolumeIntegralInitial);
    free(solver->TotalBoundaryIntegral);
    free(solver->ConservationError);
    free(solver->stride_with_ghosts);
    free(solver->stride_without_ghosts);

    /* uC/fluxC/Deriv1/Deriv2 are freed above (GPUFree or free depending on build/runtime) */

    if (solver->filename_index) free(solver->filename_index);

  }

  return(0);
}
