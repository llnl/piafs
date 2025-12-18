/*! @file gpu_time_integration.c
    @brief GPU-aware time integration support
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gpu.h>
#include <gpu_mpi.h>
#include <gpu_hyperbolic_function.h>  /* Now in src/HyParFunctions/ */
#include <gpu_parabolic_function.h>    /* Now in src/PhysicalModels/NavierStokes3D/ */
#include <gpu_arrayfunctions.h>
#include <gpu_runtime.h>
#include <timeintegration.h>
#include <simulation_object.h>
#include <hypar.h>
#include <mpivars.h>

/* GPU-aware time pre-step function
   Ensures boundary conditions and MPI exchange use GPU arrays
*/
int GPUTimePreStep(void *ts)
{
  TimeIntegration *TS = (TimeIntegration*) ts;
  SimulationObject *sim = (SimulationObject*) TS->simulation;
  int ns, nsims = TS->nsims;

  for (ns = 0; ns < nsims; ns++) {
    HyPar *solver = &(sim[ns].solver);
    MPIVariables *mpi = &(sim[ns].mpi);
    double *u = solver->u;

    if (GPUShouldUse()) {
      /* Apply boundary conditions (may need GPU-aware version) */
      solver->ApplyBoundaryConditions(solver, mpi, u, NULL, TS->waqt);

      /* Use GPU-aware MPI exchange */
      GPUMPIExchangeBoundariesnD(
        solver->ndims,
        solver->nvars,
        solver->dim_local,
        solver->ghosts,
        mpi,
        u
      );
    } else {
      /* CPU path - use original functions */
      solver->ApplyBoundaryConditions(solver, mpi, u, NULL, TS->waqt);
      MPIExchangeBoundariesnD(
        solver->ndims,
        solver->nvars,
        solver->dim_local,
        solver->ghosts,
        mpi,
        u
      );
    }
  }

  return 0;
}

/* GPU-aware RHS function
   Uses GPU functions for hyperbolic and parabolic terms
   Note: u and rhs are now GPU arrays (TS->U[stage] and TS->Udot[stage])
*/
int GPUTimeRHSFunctionExplicit(
  double *rhs,
  double *u,
  void *s,
  void *m,
  double t
)
{
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;

  if (GPUShouldUse()) {
    const int nvars = solver->nvars;
    const int npoints_wghosts = solver->npoints_local_wghosts;
    int size = npoints_wghosts * nvars;
    
    /* Safety check: ensure arrays are valid */
    if (!rhs || !u || !solver->hyp || !solver->par) {
      fprintf(stderr, "Error: GPUTimeRHSFunctionExplicit: NULL pointer detected\n");
      fprintf(stderr, "  rhs=%p, u=%p, solver->hyp=%p, solver->par=%p\n",
              rhs, u, solver->hyp, solver->par);
      return 1;
    }

    /* IMPORTANT:
       Do NOT copy stage state into solver->u.
       In RK, solver->u must remain the base (time level n) state while stage states live in TS->U[stage].
       Copying u -> solver->u corrupts subsequent stages. */

    /* Match CPU RHS behavior: apply BCs + exchange ghost points BEFORE spatial ops.
       CPU TimeRHSFunctionExplicit does this every time RHS is evaluated (including RK stages). */
    if (solver->ApplyBoundaryConditions) {
      int bc_ierr = solver->ApplyBoundaryConditions(solver, mpi, u, NULL, t);
      if (bc_ierr) return bc_ierr;
    }
    {
      int ex_ierr = GPUMPIExchangeBoundariesnD(
        solver->ndims,
        solver->nvars,
        solver->dim_local,
        solver->ghosts,
        mpi,
        u
      );
      if (ex_ierr) return ex_ierr;
    }

    /* Validate u before computing RHS (INTERIOR ONLY; ghosts may be unset at corners) */
    if (GPUShouldValidate()) {
      const int ndims = solver->ndims;
      const int ghosts = solver->ghosts;
      const int *dim = solver->dim_local;
      const int *stride = solver->stride_with_ghosts; /* host pointer */

      double *u_host_check = (double*) malloc(((size_t)npoints_wghosts * (size_t)nvars) * sizeof(double));
      if (u_host_check) {
        GPUCopyToHost(u_host_check, u, ((size_t)npoints_wghosts * (size_t)nvars) * sizeof(double));
        GPUSync();

        for (int p = 0; p < npoints_wghosts; p++) {
          /* decode multi-index to decide if p is an interior cell */
          int tmp = p;
          int is_interior = 1;
          for (int d = ndims - 1; d >= 0; d--) {
            const int id = tmp / stride[d];
            tmp -= id * stride[d];
            if (id < ghosts || id >= (ghosts + dim[d])) is_interior = 0;
          }
          if (!is_interior) continue;

          /* check all vars for NaN/Inf on interior */
          for (int v = 0; v < nvars; v++) {
            const double val = u_host_check[p*nvars + v];
            if (isnan(val) || isinf(val)) {
              fprintf(stderr,
                      "Error: NaN/Inf detected in solver->u[p=%d,v=%d]=%e before HyperbolicFunction (t=%e)\n",
                      p, v, val, t);
              free(u_host_check);
              return 1;
            }
          }

          /* physics check on interior density */
          const double rho = u_host_check[p*nvars + 0];
          if (!(rho > 0.0)) {
            /* decode indices for diagnostics */
            int tmp2 = p;
            int idx3[3] = {0,0,0};
            for (int d = ndims - 1; d >= 0; d--) {
              const int id = tmp2 / stride[d];
              tmp2 -= id * stride[d];
              if (d < 3) idx3[d] = id;
            }
            fprintf(stderr,
                    "Error: non-positive density detected before HyperbolicFunction: rho=%e at point %d idx=[%d,%d,%d] (t=%e)\n",
                    rho, p, idx3[0], idx3[1], idx3[2], t);
            free(u_host_check);
            return 1;
          }
        }

        free(u_host_check);
      }
    }

    /* Initialize RHS to zero on GPU */
    GPUArraySetValue(rhs, 0.0, size);

    /* Compute hyperbolic term using GPU */
    if (solver->HyperbolicFunction) {
      /* GPUHyperbolicFunction computes into solver->hyp */
      int hyp_ierr = GPUHyperbolicFunction(
        solver->hyp, u, solver, mpi, t, 1,
        solver->FFunction,
        solver->Upwind
      );
      if (hyp_ierr) return hyp_ierr;
      
      /* Validate hyp after computation */
      if (GPUShouldValidate()) {
        double *hyp_host_check = (double*) malloc(size * sizeof(double));
        if (hyp_host_check) {
          GPUCopyToHost(hyp_host_check, solver->hyp, size * sizeof(double));
          GPUSync();
          for (int i = 0; i < size; i++) {
            if (isnan(hyp_host_check[i]) || isinf(hyp_host_check[i])) {
              fprintf(stderr, "Error: NaN/Inf detected in solver->hyp[%d]=%e after HyperbolicFunction\n",
                      i, hyp_host_check[i]);
              free(hyp_host_check);
              return 1;
            }
          }
          free(hyp_host_check);
        }
      }
      
      /* Add to RHS (negate for hyperbolic term) */
      GPUArrayAXPY(solver->hyp, -1.0, rhs, size);
    }

    /* Compute parabolic term using GPU */
    if (solver->ParabolicFunction) {
      /* ParabolicFunction is already set to GPU version if GPU is enabled */
      int par_ierr = solver->ParabolicFunction(
        solver->par, u, solver, mpi, t
      );
      if (par_ierr) return par_ierr;
      /* Add parabolic term to RHS on GPU */
      GPUArrayAdd(rhs, rhs, solver->par, size);
    }

    /* Compute source term */
    if (solver->SourceFunction) {
      /* SourceFunction is already set to GPU version if GPU is enabled */
      int sou_ierr = solver->SourceFunction(solver->source, u, solver, mpi, t);
      if (sou_ierr) return sou_ierr;
      
      /* Add source term to RHS on GPU */
      GPUArrayAdd(rhs, rhs, solver->source, size);
    }
    /* Avoid forced device sync on the hot path (hurts performance).
       Enable per-op sync only for debugging via PIAFS_GPU_SYNC_EVERY_OP=1. */
    if (GPUShouldSyncEveryOp()) GPUSync();
    return 0;
  } else {
    /* CPU fallback - use original RHS function */
    return 1;
  }
}

/* Initialize GPU arrays for time integration stages */
int GPUInitializeTimeIntegrationArrays(void *ts)
{
  TimeIntegration *TS = (TimeIntegration*) ts;
  SimulationObject *sim = (SimulationObject*) TS->simulation;

  if (!GPUShouldUse()) {
    return 0; /* Nothing to do */
  }

  /* Allocate stage arrays on GPU if using Runge-Kutta */
  if (TS->U && TS->Udot) {
    int nstages = 0;
    while (TS->U[nstages] != NULL && nstages < 10) nstages++; /* Count stages */

    for (int i = 0; i < nstages; i++) {
      if (TS->U[i]) {
        /* Check if already on GPU */
        /* For now, assume they're allocated on host and need to be on GPU */
        /* In practice, these should be allocated on GPU from the start */
      }
    }
  }

  return 0;
}

