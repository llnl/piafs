// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file InitialSolution.c
    @author Debojyoti Ghosh
    @brief Read in initial solution from file
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <basic.h>
#include <common.h>
#include <arrayfunctions.h>
#include <io.h>
#include <mpivars.h>
#include <simulation_object.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_mpi.h>
#include <gpu_initialize.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_mpi.h>
#include <gpu_initialize.h>
#endif

int VolumeIntegral(double*,double*,void*,void*);

/*! Read in initial solution from file, and compute grid spacing
    and volume integral of the initial solution */
int InitialSolution ( void  *s,   /*!< Array of simulation objects of type #SimulationObject */
                      int   nsims /*!< Number of simulation objects */
                    )
{
  SimulationObject* simobj = (SimulationObject*) s;
  int n, flag, d, i, offset, ierr;

  for (n = 0; n < nsims; n++) {

    int ghosts = simobj[n].solver.ghosts;

    char fname_root[_MAX_STRING_SIZE_] = "initial";
    if (nsims > 1) {
      char index[_MAX_STRING_SIZE_];
      GetStringFromInteger(n, index, (int)log10(nsims)+1);
      strcat(fname_root, "_");
      strcat(fname_root, index);
    }

    /* If GPU is enabled, read u to host buffer first, then copy to GPU. x stays on host. */
#if defined(GPU_CUDA) || defined(GPU_HIP)
    if (GPUShouldUse()) {
      /* Allocate temporary host buffer for u */
      int size_u = simobj[n].solver.npoints_local_wghosts * simobj[n].solver.nvars;
      double *u_host = (double*) calloc(size_u, sizeof(double));
      
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for initial solution\n");
        return 1;
      }
      
      /* Read to host buffers (x is already on host) */
      ierr = ReadArray( simobj[n].solver.ndims,
                        simobj[n].solver.nvars,
                        simobj[n].solver.dim_global,
                        simobj[n].solver.dim_local,
                        simobj[n].solver.ghosts,
                        &(simobj[n].solver),
                        &(simobj[n].mpi),
                        simobj[n].solver.x,
                        u_host,
                        fname_root,
                        &flag );
      if (ierr) {
        fprintf(stderr, "Error in InitialSolution() on rank %d.\n",
                simobj[n].mpi.rank);
        free(u_host);
        return ierr;
      }
      if (!flag) {
        fprintf(stderr,"Error: initial solution file not found.\n");
        free(u_host);
        return(1);
      }
      CHECKERR(ierr);
      
      /* Exchange MPI-boundary values on host */
      MPIExchangeBoundariesnD(  simobj[n].solver.ndims,
                                simobj[n].solver.nvars,
                                simobj[n].solver.dim_local,
                                simobj[n].solver.ghosts,
                                &(simobj[n].mpi),
                                u_host  );
      
      /* Copy u from host to GPU (x will be copied later via GPUCopyGridArraysToDevice) */
      GPUCopyToDevice(simobj[n].solver.u, u_host, size_u * sizeof(double));
      GPUSync();
      
      /* Free host buffer */
      free(u_host);
      
      /* Exchange boundaries on GPU */
      GPUMPIExchangeBoundariesnD(  simobj[n].solver.ndims,
                                  simobj[n].solver.nvars,
                                  simobj[n].solver.dim_local,
                                  simobj[n].solver.ghosts,
                                  &(simobj[n].mpi),
                                  simobj[n].solver.u  );
    } else {
      ierr = ReadArray( simobj[n].solver.ndims,
                        simobj[n].solver.nvars,
                        simobj[n].solver.dim_global,
                        simobj[n].solver.dim_local,
                        simobj[n].solver.ghosts,
                        &(simobj[n].solver),
                        &(simobj[n].mpi),
                        simobj[n].solver.x,
                        simobj[n].solver.u,
                        fname_root,
                        &flag );
      if (ierr) {
        fprintf(stderr, "Error in InitialSolution() on rank %d.\n",
                simobj[n].mpi.rank);
        return ierr;
      }
      if (!flag) {
        fprintf(stderr,"Error: initial solution file not found.\n");
        return(1);
      }
      CHECKERR(ierr);

      /* exchange MPI-boundary values of u between processors */
      MPIExchangeBoundariesnD(  simobj[n].solver.ndims,
                                simobj[n].solver.nvars,
                                simobj[n].solver.dim_local,
                                simobj[n].solver.ghosts,
                                &(simobj[n].mpi),
                                simobj[n].solver.u  );
    }
#else
    ierr = ReadArray( simobj[n].solver.ndims,
                      simobj[n].solver.nvars,
                      simobj[n].solver.dim_global,
                      simobj[n].solver.dim_local,
                      simobj[n].solver.ghosts,
                      &(simobj[n].solver),
                      &(simobj[n].mpi),
                      simobj[n].solver.x,
                      simobj[n].solver.u,
                      fname_root,
                      &flag );
    if (ierr) {
      fprintf(stderr, "Error in InitialSolution() on rank %d.\n",
              simobj[n].mpi.rank);
      return ierr;
    }
    if (!flag) {
      fprintf(stderr,"Error: initial solution file not found.\n");
      return(1);
    }
    CHECKERR(ierr);

    /* exchange MPI-boundary values of u between processors */
    MPIExchangeBoundariesnD(  simobj[n].solver.ndims,
                              simobj[n].solver.nvars,
                              simobj[n].solver.dim_local,
                              simobj[n].solver.ghosts,
                              &(simobj[n].mpi),
                              simobj[n].solver.u  );
#endif

    /* calculate dxinv (always on host - x and dxinv stay on host) */
    offset = 0;
    for (d = 0; d < simobj[n].solver.ndims; d++) {
      for (i = 0; i < simobj[n].solver.dim_local[d]; i++) {
        simobj[n].solver.dxinv[i+offset+ghosts]
          = 2.0 / (simobj[n].solver.x[i+1+offset+ghosts]-simobj[n].solver.x[i-1+offset+ghosts]);
      }
      offset += (simobj[n].solver.dim_local[d] + 2*ghosts);
    }

    /* exchange MPI-boundary values of dxinv between processors */
    offset = 0;
    for (d = 0; d < simobj[n].solver.ndims; d++) {
      ierr = MPIExchangeBoundaries1D( &(simobj[n].mpi),
                                      &(simobj[n].solver.dxinv[offset]),
                                      simobj[n].solver.dim_local[d],
                                      ghosts,
                                      d,
                                      simobj[n].solver.ndims ); CHECKERR(ierr);
      if (ierr) {
        fprintf(stderr, "Error in InitialSolution() on rank %d.\n",
                simobj[n].mpi.rank);
        return ierr;
      }
      offset += (simobj[n].solver.dim_local[d] + 2*ghosts);
    }

    /* fill in ghost values of dxinv at physical boundaries by extrapolation */
    offset = 0;
    for (d = 0; d < simobj[n].solver.ndims; d++) {
      double *dxinv = &(simobj[n].solver.dxinv[offset]);
      int    *dim = simobj[n].solver.dim_local;
      if (simobj[n].mpi.ip[d] == 0) {
        /* fill left boundary along this dimension */
        for (i = 0; i < ghosts; i++) dxinv[i] = dxinv[ghosts];
      }
      if (simobj[n].mpi.ip[d] == simobj[n].mpi.iproc[d]-1) {
        /* fill right boundary along this dimension */
        for (i = dim[d]+ghosts; i < dim[d]+2*ghosts; i++) dxinv[i] = dxinv[dim[d]+ghosts-1];
      }
      offset  += (dim[d] + 2*ghosts);
    }

    /* calculate volume integral of the initial solution */
#if defined(GPU_CUDA) || defined(GPU_HIP)
    if (GPUShouldUse()) {
      /* VolumeIntegral accesses u - need host copy. dxinv is already on host */
      int size_u = simobj[n].solver.npoints_local_wghosts * simobj[n].solver.nvars;
      double *u_host = (double*) malloc(size_u * sizeof(double));
      
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for VolumeIntegral\n");
        return 1;
      }
      
      GPUCopyToHost(u_host, simobj[n].solver.u, size_u * sizeof(double));
      GPUSync();
      
      ierr = VolumeIntegral(  simobj[n].solver.VolumeIntegralInitial,
                            u_host,
                            &(simobj[n].solver),
                            &(simobj[n].mpi) ); CHECKERR(ierr);
      
      free(u_host);
    } else {
      ierr = VolumeIntegral(  simobj[n].solver.VolumeIntegralInitial,
                            simobj[n].solver.u,
                            &(simobj[n].solver),
                            &(simobj[n].mpi) ); CHECKERR(ierr);
    }
#else
    ierr = VolumeIntegral(  simobj[n].solver.VolumeIntegralInitial,
                            simobj[n].solver.u,
                            &(simobj[n].solver),
                            &(simobj[n].mpi) ); CHECKERR(ierr);
#endif
    if (ierr) {
      fprintf(stderr, "Error in InitialSolution() on rank %d.\n",
              simobj[n].mpi.rank);
      return ierr;
    }
    if (!simobj[n].mpi.rank) {
      if (nsims > 1) printf("Volume integral of the initial solution on domain %d:\n", n);
      else           printf("Volume integral of the initial solution:\n");
      for (d=0; d<simobj[n].solver.nvars; d++) {
        printf("%2d:  %1.16E\n",d,simobj[n].solver.VolumeIntegralInitial[d]);
      }
    }
    /* Set initial total boundary flux integral to zero */
    _ArraySetValue_(simobj[n].solver.TotalBoundaryIntegral,simobj[n].solver.nvars,0);

  }

  /* Copy grid arrays to device for GPU runs (one-time copy after initialization) */
#if defined(GPU_CUDA) || defined(GPU_HIP)
  if (GPUShouldUse()) {
    ierr = GPUCopyGridArraysToDevice(simobj, nsims);
    if (ierr) {
      fprintf(stderr, "Error: Failed to copy grid arrays to GPU\n");
      return ierr;
    }
  }
#endif

  return 0;
}
