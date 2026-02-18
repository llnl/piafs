// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file ApplyBoundaryConditions.c
 *  @author Debojyoti Ghosh
 *  @brief Apply physical boundary conditions to domain.
 *
 *  Contains the function that applies the physical boundary conditions
 *  to each boundary zone.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <mpivars.h>
#include <boundaryconditions.h>
#include <hypar.h>
#if defined(GPU_CUDA) || defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_arrayfunctions.h>
#include <gpu_bc.h>
#endif

/*!
 * \brief Applies the boundary conditions specified for each boundary zone.
 *
 * The solver object (of type #HyPar) contains an oject of type #DomainBoundary
 * that contains all the boundary information (dimension, extent, face, type, etc).
 * This function iterates through each of the boundary zones
 * (#HyPar::boundary[#HyPar::nBoundaryZones]) and calls the corresponding boundary
 * condition function.
 * \n\n
 * The variable \a flag indicates if the array \a x is the solution, or a delta-solution
 * (from implicit time-integration methods).
*/
int ApplyBoundaryConditions(void    *s,     /*!< Object of type #HyPar containing solver-related variables */
                            void    *m,     /*!< Object of type #MPIVariables containing MPI-related variables */
                            double  *x,     /*!< The solution vector on which the boundary conditions are to be applied */
                            double  *xref,  /*!< Reference solution vector, if needed */
                            double  waqt    /*!< Current simulation time */
                           )
{
  HyPar           *solver   = (HyPar*)          s;
  DomainBoundary  *boundary = (DomainBoundary*) solver->boundary;
  MPIVariables    *mpi      = (MPIVariables*)   m;
  int             nb        = solver->nBoundaryZones;

  /* Safety checks */
  if (!solver) {
    fprintf(stderr, "Error: ApplyBoundaryConditions: solver is NULL\n");
    return 1;
  }
  if (!boundary && nb > 0) {
    fprintf(stderr, "Error: ApplyBoundaryConditions: boundary is NULL but nBoundaryZones=%d\n", nb);
    return 1;
  }
  if (!x) {
    fprintf(stderr, "Error: ApplyBoundaryConditions: x is NULL\n");
    return 1;
  }

  int* dim_local;
  dim_local = solver->dim_local;

  /* Apply domain boundary conditions to x */
#if defined(GPU_CUDA) || defined(GPU_HIP)
  if (GPUShouldUse()) {
#ifndef serial
    /* In MPI runs with purely periodic physical boundaries where all dimensions
       have more than 1 process, halo exchange handles periodicity.
       But if any dimension has only 1 process, we need to apply the periodic BC
       kernel for that dimension since MPI exchange won't handle it. */
    int all_periodic_and_multi_proc = 1;
    for (int n = 0; n < nb; n++) {
      if (strcmp(boundary[n].bctype, _PERIODIC_)) {
        all_periodic_and_multi_proc = 0;
        break;
      }
      /* Check if this periodic boundary needs local handling */
      if (mpi->iproc[boundary[n].dim] == 1) {
        all_periodic_and_multi_proc = 0;
      }
    }
    if (all_periodic_and_multi_proc) return 0;
#endif
    
    /* Apply boundary conditions directly on GPU using native kernels */
    int n;
    for (n = 0; n < nb; n++) {
      if (!boundary[n].on_this_proc) continue;
      
      DomainBoundary *bc = &boundary[n];
      int bc_dim = bc->dim;
      int bc_face = bc->face;
      
      /* Check if this BC type has a GPU implementation */
      int gpu_bc_handled = 0;
      
      if (!strcmp(bc->bctype, _PERIODIC_)) {
#ifdef serial
        /* Single-processor periodic: use GPU kernel */
        gpu_launch_bc_periodic(x, solver->nvars, solver->ndims, dim_local,
                               solver->ghosts, bc_dim, bc_face, bc->is, bc->ie, 256);
        GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
        gpu_bc_handled = 1;
#else
        /* Multi-processor: handled by MPI exchange, except for single-process dimensions */
        if (mpi->iproc[bc_dim] == 1) {
          /* Single-proc dimension needs explicit periodic BC */
          gpu_launch_bc_periodic(x, solver->nvars, solver->ndims, dim_local,
                                 solver->ghosts, bc_dim, bc_face, bc->is, bc->ie, 256);
          GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
        }
        gpu_bc_handled = 1;
#endif
      } else if (!strcmp(bc->bctype, _EXTRAPOLATE_)) {
        gpu_launch_bc_extrapolate(x, solver->nvars, solver->ndims, dim_local,
                                  solver->ghosts, bc_dim, bc_face, bc->is, bc->ie, 256);
        GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _DIRICHLET_)) {
        gpu_launch_bc_dirichlet(x, bc->DirichletValue, solver->nvars, solver->ndims,
                                dim_local, solver->ghosts, bc->is, bc->ie, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _REFLECT_)) {
        gpu_launch_bc_reflect(x, solver->nvars, solver->ndims, dim_local,
                              solver->ghosts, bc_dim, bc_face, bc->is, bc->ie, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _SLIP_WALL_)) {
        gpu_launch_bc_slipwall(x, solver->nvars, solver->ndims, dim_local,
                               solver->ghosts, bc_dim, bc_face, bc->is, bc->ie, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _NOSLIP_WALL_)) {
        double wall_u = bc->FlowVelocity ? bc->FlowVelocity[0] : 0.0;
        double wall_v = (bc->FlowVelocity && solver->ndims >= 2) ? bc->FlowVelocity[1] : 0.0;
        double wall_w = (bc->FlowVelocity && solver->ndims >= 3) ? bc->FlowVelocity[2] : 0.0;
        gpu_launch_bc_noslipwall(x, solver->nvars, solver->ndims, dim_local,
                                 solver->ghosts, bc_dim, bc_face, bc->is, bc->ie,
                                 bc->gamma, wall_u, wall_v, wall_w, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _SUPERSONIC_OUTFLOW_)) {
        gpu_launch_bc_supersonic_outflow(x, solver->nvars, solver->ndims, dim_local,
                                         solver->ghosts, bc_dim, bc_face, bc->is, bc->ie, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _SUPERSONIC_INFLOW_)) {
        double u = bc->FlowVelocity ? bc->FlowVelocity[0] : 0.0;
        double v = (bc->FlowVelocity && solver->ndims >= 2) ? bc->FlowVelocity[1] : 0.0;
        double w = (bc->FlowVelocity && solver->ndims >= 3) ? bc->FlowVelocity[2] : 0.0;
        gpu_launch_bc_supersonic_inflow(x, solver->nvars, solver->ndims, dim_local,
                                        solver->ghosts, bc->is, bc->ie,
                                        bc->gamma, bc->FlowDensity, u, v, w, bc->FlowPressure, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _SUBSONIC_INFLOW_)) {
        double u = bc->FlowVelocity ? bc->FlowVelocity[0] : 0.0;
        double v = (bc->FlowVelocity && solver->ndims >= 2) ? bc->FlowVelocity[1] : 0.0;
        double w = (bc->FlowVelocity && solver->ndims >= 3) ? bc->FlowVelocity[2] : 0.0;
        gpu_launch_bc_subsonic_inflow(x, solver->nvars, solver->ndims, dim_local,
                                      solver->ghosts, bc_dim, bc_face, bc->is, bc->ie,
                                      bc->gamma, bc->FlowDensity, u, v, w, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _SUBSONIC_OUTFLOW_)) {
        gpu_launch_bc_subsonic_outflow(x, solver->nvars, solver->ndims, dim_local,
                                       solver->ghosts, bc_dim, bc_face, bc->is, bc->ie,
                                       bc->gamma, bc->FlowPressure, 256);
        gpu_bc_handled = 1;
      } else if (!strcmp(bc->bctype, _SUBSONIC_AMBIVALENT_)) {
        double u = bc->FlowVelocity ? bc->FlowVelocity[0] : 0.0;
        double v = (bc->FlowVelocity && solver->ndims >= 2) ? bc->FlowVelocity[1] : 0.0;
        double w = (bc->FlowVelocity && solver->ndims >= 3) ? bc->FlowVelocity[2] : 0.0;
        gpu_launch_bc_subsonic_ambivalent(x, solver->nvars, solver->ndims, dim_local,
                                          solver->ghosts, bc_dim, bc_face, bc->is, bc->ie,
                                          bc->gamma, bc->FlowDensity, u, v, w, bc->FlowPressure, 256);
        gpu_bc_handled = 1;
      }

      /* If BC type not handled by GPU kernel, fall back to host staging */
      if (!gpu_bc_handled) {
        if (!bc->BCFunctionU) {
          fprintf(stderr, "Error: ApplyBoundaryConditions: BCFunctionU is NULL for boundary %d\n", n);
          return 1;
        }
        /* Need to stage to host for this BC */
        int size = solver->npoints_local_wghosts * solver->nvars;
        double *x_host = (double*) malloc(size * sizeof(double));
        if (!x_host) {
          fprintf(stderr, "Error: Failed to allocate host buffer for boundary conditions\n");
          return 1;
        }
        GPUCopyToHost(x_host, x, size * sizeof(double));
        bc->BCFunctionU(bc, mpi, solver->ndims, solver->nvars,
                        dim_local, solver->ghosts, x_host, waqt);
        GPUCopyToDevice(x, x_host, size * sizeof(double));
        free(x_host);
      }
    }
    if (GPUShouldSyncEveryOp()) GPUSync();
  } else {
    int n;
    for (n = 0; n < nb; n++) {
      if (!boundary[n].BCFunctionU) {
        fprintf(stderr, "Error: ApplyBoundaryConditions: BCFunctionU is NULL for boundary %d\n", n);
        return 1;
      }
      boundary[n].BCFunctionU(&boundary[n],mpi,solver->ndims,solver->nvars,
                              dim_local,solver->ghosts,x,waqt);
    }
  }
#else
  int n;
  for (n = 0; n < nb; n++) {
    if (!boundary[n].BCFunctionU) {
      fprintf(stderr, "Error: ApplyBoundaryConditions: BCFunctionU is NULL for boundary %d\n", n);
      return 1;
    }
    boundary[n].BCFunctionU(&boundary[n],mpi,solver->ndims,solver->nvars,
                            dim_local,solver->ghosts,x,waqt);
  }
#endif

  return(0);
}
