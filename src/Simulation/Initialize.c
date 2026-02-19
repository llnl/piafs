// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file Initialize.c
    @author Debojyoti Ghosh
    @brief Initialization function
*/

#include <stdio.h>
#include <stdlib.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <mpivars.h>
#include <simulation_object.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_initialize.h>
#include <gpu_runtime.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_initialize.h>
#include <gpu_runtime.h>
#endif

/*! Initialization function called at the beginning of a simulation. This function
    does the following:
    + allocates memory for MPI related arrays
    + initializes the values for MPI variables
    + creates sub-communicators and communication groups
    + allocates memory for arrays to store solution, right-hand-side,
      flux, and other working vectors.
    + initializes function counters to zero
*/
int Initialize( void *s,    /*!< Array of simulation objects of type #SimulationObject */
                int  nsims  /*!< Number of simulation objects */
              )
{
  SimulationObject* simobj = (SimulationObject*) s;
  int i,d,n;

  if (nsims == 0) {
    return 1;
  }

  if (!simobj[0].mpi.rank)  printf("Partitioning domain and allocating data arrays.\n");

  for (n = 0; n < nsims; n++) {

    /* this is a full initialization, not a barebones one */
    simobj[n].is_barebones = 0;

    /* allocations */
    simobj[n].mpi.ip           = (int*) calloc (simobj[n].solver.ndims,sizeof(int));
    simobj[n].mpi.is           = (int*) calloc (simobj[n].solver.ndims,sizeof(int));
    simobj[n].mpi.ie           = (int*) calloc (simobj[n].solver.ndims,sizeof(int));
    simobj[n].mpi.bcperiodic   = (int*) calloc (simobj[n].solver.ndims,sizeof(int));
    simobj[n].solver.dim_local = (int*) calloc (simobj[n].solver.ndims,sizeof(int));
    simobj[n].solver.isPeriodic= (int*) calloc (simobj[n].solver.ndims,sizeof(int));

#ifndef serial
    _DECLARE_IERR_;

    /* Domain partitioning */
    int total_proc = 1;
    for (i=0; i<simobj[n].solver.ndims; i++) total_proc *= simobj[n].mpi.iproc[i];
    if (simobj[n].mpi.nproc != total_proc) {
      fprintf(stderr,"Error on rank %d: total number of processes is not consistent ", simobj[n].mpi.rank);
      fprintf(stderr,"with number of processes along each dimension.\n");
      if (nsims > 1) fprintf(stderr,"for domain %d.\n", n);
      fprintf(stderr,"mpiexec was called with %d processes, ",simobj[n].mpi.nproc);
      fprintf(stderr,"total number of processes from \"solver.inp\" is %d.\n", total_proc);
      return(1);
    }

    /* calculate ndims-D rank of each process (ip[]) from rank in MPI_COMM_WORLD */
    IERR MPIRanknD( simobj[n].solver.ndims,
                    simobj[n].mpi.rank,
                    simobj[n].mpi.iproc,
                    simobj[n].mpi.ip); CHECKERR(ierr);

    /* calculate local domain sizes along each dimension */
    for (i=0; i<simobj[n].solver.ndims; i++) {
      simobj[n].solver.dim_local[i] = MPIPartition1D( simobj[n].solver.dim_global[i],
                                                      simobj[n].mpi.iproc[i],
                                                      simobj[n].mpi.ip[i] );
    }

    /* calculate local domain limits in terms of global domain */
    IERR MPILocalDomainLimits(  simobj[n].solver.ndims,
                                simobj[n].mpi.rank,
                                &(simobj[n].mpi),
                                simobj[n].solver.dim_global,
                                simobj[n].mpi.is,
                                simobj[n].mpi.ie  );
    CHECKERR(ierr);

    /* create sub-communicators for parallel computations along grid lines in each dimension */
    IERR MPICreateCommunicators(simobj[n].solver.ndims,&(simobj[n].mpi)); CHECKERR(ierr);

    /* initialize periodic BC flags to zero */
    for (i=0; i<simobj[n].solver.ndims; i++) simobj[n].mpi.bcperiodic[i] = 0;

    /* create communication groups */
    IERR MPICreateIOGroups(&(simobj[n].mpi)); CHECKERR(ierr);

#else

    for (i=0; i<simobj[n].solver.ndims; i++) {
      simobj[n].mpi.ip[i]            = 0;
      simobj[n].solver.dim_local[i]  = simobj[n].solver.dim_global[i];
      simobj[n].mpi.iproc[i]         = 1;
      simobj[n].mpi.is[i]            = 0;
      simobj[n].mpi.ie[i]            = simobj[n].solver.dim_local[i];
      simobj[n].mpi.bcperiodic[i]    = 0;
    }

#endif

    simobj[n].solver.npoints_global
      = simobj[n].solver.npoints_local
      = simobj[n].solver.npoints_local_wghosts
      = 1;
    for (i=0; i<simobj[n].solver.ndims; i++) {
      simobj[n].solver.npoints_global *= simobj[n].solver.dim_global[i];
      simobj[n].solver.npoints_local *= simobj[n].solver.dim_local [i];
      simobj[n].solver.npoints_local_wghosts *= (simobj[n].solver.dim_local[i]+2*simobj[n].solver.ghosts);
    }

    /* Allocations */
    simobj[n].solver.index = (int*) calloc ((short)simobj[n].solver.ndims,sizeof(int));
    simobj[n].solver.stride_with_ghosts = (int*) calloc ((short)simobj[n].solver.ndims,sizeof(int));
    simobj[n].solver.stride_without_ghosts = (int*) calloc ((short)simobj[n].solver.ndims,sizeof(int));
    int accu1 = 1, accu2 = 1;
    for (i=0; i<simobj[n].solver.ndims; i++) {
      simobj[n].solver.stride_with_ghosts[i]    = accu1;
      simobj[n].solver.stride_without_ghosts[i] = accu2;
      accu1 *= (simobj[n].solver.dim_local[i]+2*simobj[n].solver.ghosts);
      accu2 *=  simobj[n].solver.dim_local[i];
    }

    /* state variables */
    int size = 1;
    for (i=0; i<simobj[n].solver.ndims; i++) {
      size *= (simobj[n].solver.dim_local[i]+2*simobj[n].solver.ghosts);
    }
    simobj[n].solver.ndof_cells_wghosts = simobj[n].solver.nvars*size;

    /* Allocate on GPU if available, otherwise on host */
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      if (GPUAllocateSolutionArrays(&simobj[n], 1)) {
        fprintf(stderr, "Error: GPU allocation failed, falling back to CPU\n");
        /* Fall back to CPU allocation */
        simobj[n].solver.u = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
        simobj[n].solver.hyp = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
        simobj[n].solver.par = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
        simobj[n].solver.source = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      }
    } else {
      /* CPU allocation */
      simobj[n].solver.u = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.hyp = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.par = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.source = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      if (GPUAllocateSolutionArrays(&simobj[n], 1)) {
        fprintf(stderr, "Error: GPU allocation failed, falling back to CPU\n");
        /* Fall back to CPU allocation */
        simobj[n].solver.u = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
        simobj[n].solver.hyp = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
        simobj[n].solver.par = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
        simobj[n].solver.source = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      }
    } else {
      /* CPU allocation */
      simobj[n].solver.u = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.hyp = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.par = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.source = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    }
#else
    /* CPU allocation */
    simobj[n].solver.u = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    simobj[n].solver.hyp = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    simobj[n].solver.par = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    simobj[n].solver.source = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
#endif

    /* grid */
    size = 0;
    for (i=0; i<simobj[n].solver.ndims; i++) {
      size += (simobj[n].solver.dim_local[i]+2*simobj[n].solver.ghosts);
    }
    simobj[n].solver.size_x = size;

    /* Always allocate grid arrays on CPU (they don't change and are accessed by CPU code) */
    simobj[n].solver.x = (double*) calloc (size,sizeof(double));
    simobj[n].solver.dxinv = (double*) calloc (size,sizeof(double));

    /* Initialize device pointers to NULL */
    simobj[n].solver.d_x = NULL;
    simobj[n].solver.d_dxinv = NULL;

    /* Allocate device copies for GPU builds (needed by GPU kernels like CFL computation) */
#if defined(GPU_CUDA) || defined(GPU_HIP)
    if (GPUShouldUse()) {
      if (GPUAllocate((void**)&simobj[n].solver.d_x, size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate d_x on GPU\n");
        return 1;
      }
      if (GPUAllocate((void**)&simobj[n].solver.d_dxinv, size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate d_dxinv on GPU\n");
        GPUFree(simobj[n].solver.d_x);
        return 1;
      }
    }
#endif

    /* cell-centered arrays needed to compute fluxes */
    size = 1;
    for (i=0; i<simobj[n].solver.ndims; i++) {
      size *= (simobj[n].solver.dim_local[i]+2*simobj[n].solver.ghosts);
    }

#ifdef GPU_CUDA
    if (GPUShouldUse() && simobj[n].solver.u) {
      /* Already allocated in GPUAllocateSolutionArrays */
      /* Just verify they exist */
      if (!simobj[n].solver.uC || !simobj[n].solver.fluxC ||
          !simobj[n].solver.Deriv1 || !simobj[n].solver.Deriv2) {
        fprintf(stderr, "Error: GPU cell-centered arrays not properly allocated\n");
        return 1;
      }
    } else {
      simobj[n].solver.uC = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.fluxC = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.Deriv1 = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.Deriv2 = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse() && simobj[n].solver.u) {
      /* Already allocated in GPUAllocateSolutionArrays */
      /* Just verify they exist */
      if (!simobj[n].solver.uC || !simobj[n].solver.fluxC ||
          !simobj[n].solver.Deriv1 || !simobj[n].solver.Deriv2) {
        fprintf(stderr, "Error: GPU cell-centered arrays not properly allocated\n");
        return 1;
      }
    } else {
      simobj[n].solver.uC = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.fluxC = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.Deriv1 = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
      simobj[n].solver.Deriv2 = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    }
#else
    simobj[n].solver.uC = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    simobj[n].solver.fluxC = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    simobj[n].solver.Deriv1 = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
    simobj[n].solver.Deriv2 = (double*) calloc (simobj[n].solver.nvars*size,sizeof(double));
#endif

    /* node-centered arrays needed to compute fluxes */
    size = 1;  for (i=0; i<simobj[n].solver.ndims; i++) size *= (simobj[n].solver.dim_local[i]+1);
    size *= simobj[n].solver.nvars;
    simobj[n].solver.ndof_nodes = size;

#ifdef GPU_CUDA
    if (GPUShouldUse() && simobj[n].solver.u) {
      /* Already allocated in GPUAllocateSolutionArrays */
      if (!simobj[n].solver.fluxI || !simobj[n].solver.uL ||
          !simobj[n].solver.uR || !simobj[n].solver.fL || !simobj[n].solver.fR) {
        fprintf(stderr, "Error: GPU interface arrays not properly allocated\n");
        return 1;
      }
    } else {
      simobj[n].solver.fluxI = (double*) calloc (size,sizeof(double));
      simobj[n].solver.uL = (double*) calloc (size,sizeof(double));
      simobj[n].solver.uR = (double*) calloc (size,sizeof(double));
      simobj[n].solver.fL = (double*) calloc (size,sizeof(double));
      simobj[n].solver.fR = (double*) calloc (size,sizeof(double));
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse() && simobj[n].solver.u) {
      /* Already allocated in GPUAllocateSolutionArrays */
      if (!simobj[n].solver.fluxI || !simobj[n].solver.uL ||
          !simobj[n].solver.uR || !simobj[n].solver.fL || !simobj[n].solver.fR) {
        fprintf(stderr, "Error: GPU interface arrays not properly allocated\n");
        return 1;
      }
    } else {
      simobj[n].solver.fluxI = (double*) calloc (size,sizeof(double));
      simobj[n].solver.uL = (double*) calloc (size,sizeof(double));
      simobj[n].solver.uR = (double*) calloc (size,sizeof(double));
      simobj[n].solver.fL = (double*) calloc (size,sizeof(double));
      simobj[n].solver.fR = (double*) calloc (size,sizeof(double));
    }
#else
    simobj[n].solver.fluxI = (double*) calloc (size,sizeof(double));
    simobj[n].solver.uL = (double*) calloc (size,sizeof(double));
    simobj[n].solver.uR = (double*) calloc (size,sizeof(double));
    simobj[n].solver.fL = (double*) calloc (size,sizeof(double));
    simobj[n].solver.fR = (double*) calloc (size,sizeof(double));
#endif

    /* allocate MPI send/receive buffer arrays */
    int bufdim[simobj[n].solver.ndims], maxbuf = 0;
    for (d = 0; d < simobj[n].solver.ndims; d++) {
      bufdim[d] = 1;
      for (i = 0; i < simobj[n].solver.ndims; i++) {
        if (i == d) bufdim[d] *= simobj[n].solver.ghosts;
        else        bufdim[d] *= simobj[n].solver.dim_local[i];
      }
      if (bufdim[d] > maxbuf) maxbuf = bufdim[d];
    }
    maxbuf *= (simobj[n].solver.nvars*simobj[n].solver.ndims);
    simobj[n].mpi.maxbuf  = maxbuf;
    size_t mpi_buf_size = (size_t)(2*simobj[n].solver.ndims*maxbuf) * sizeof(double);
#if defined(GPU_CUDA) || defined(GPU_HIP)
    /* Use pinned memory for faster GPU-CPU transfers */
    if (GPUShouldUse()) {
      simobj[n].mpi.use_gpu_pinned = 1;
      if (GPUAllocatePinned((void**)&simobj[n].mpi.sendbuf, mpi_buf_size)) {
        fprintf(stderr, "Error: Failed to allocate pinned memory for MPI sendbuf\n");
        return 1;
      }
      if (GPUAllocatePinned((void**)&simobj[n].mpi.recvbuf, mpi_buf_size)) {
        fprintf(stderr, "Error: Failed to allocate pinned memory for MPI recvbuf\n");
        GPUFreePinned(simobj[n].mpi.sendbuf);
        return 1;
      }
      memset(simobj[n].mpi.sendbuf, 0, mpi_buf_size);
      memset(simobj[n].mpi.recvbuf, 0, mpi_buf_size);
    } else
#endif
    {
      simobj[n].mpi.use_gpu_pinned = 0;
      simobj[n].mpi.sendbuf = (double*) calloc (2*simobj[n].solver.ndims*maxbuf,sizeof(double));
      simobj[n].mpi.recvbuf = (double*) calloc (2*simobj[n].solver.ndims*maxbuf,sizeof(double));
    }

    /* allocate the volume and boundary integral arrays */
    simobj[n].solver.VolumeIntegral        = (double*) calloc (simobj[n].solver.nvars  ,sizeof(double));
    simobj[n].solver.VolumeIntegralInitial = (double*) calloc (simobj[n].solver.nvars  ,sizeof(double));
    simobj[n].solver.TotalBoundaryIntegral = (double*) calloc (simobj[n].solver.nvars,sizeof(double));
    simobj[n].solver.ConservationError     = (double*) calloc (simobj[n].solver.nvars,sizeof(double));
    for (i=0; i<simobj[n].solver.nvars; i++) simobj[n].solver.ConservationError[i] = -1;

    /* StageBoundaryIntegral and StepBoundaryIntegral need to be on GPU if GPU is enabled */
    int bf_size = 2*simobj[n].solver.ndims*simobj[n].solver.nvars;
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      if (GPUAllocate((void**)&simobj[n].solver.StageBoundaryIntegral, bf_size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate StageBoundaryIntegral on GPU\n");
        return 1;
      }
      if (GPUAllocate((void**)&simobj[n].solver.StepBoundaryIntegral, bf_size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate StepBoundaryIntegral on GPU\n");
        GPUFree(simobj[n].solver.StageBoundaryIntegral);
        return 1;
      }
      /* Initialize to zero */
      if (GPUMemset(simobj[n].solver.StageBoundaryIntegral, 0, bf_size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to memset StageBoundaryIntegral\n");
        return 1;
      }
      if (GPUMemset(simobj[n].solver.StepBoundaryIntegral, 0, bf_size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to memset StepBoundaryIntegral\n");
        return 1;
      }
    } else {
      simobj[n].solver.StageBoundaryIntegral = (double*) calloc (bf_size,sizeof(double));
      simobj[n].solver.StepBoundaryIntegral  = (double*) calloc (bf_size,sizeof(double));
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      if (GPUAllocate((void**)&simobj[n].solver.StageBoundaryIntegral, bf_size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate StageBoundaryIntegral on GPU\n");
        return 1;
      }
      if (GPUAllocate((void**)&simobj[n].solver.StepBoundaryIntegral, bf_size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate StepBoundaryIntegral on GPU\n");
        GPUFree(simobj[n].solver.StageBoundaryIntegral);
        return 1;
      }
      /* Initialize to zero */
      GPUMemset(simobj[n].solver.StageBoundaryIntegral, 0, bf_size * sizeof(double));
      GPUMemset(simobj[n].solver.StepBoundaryIntegral, 0, bf_size * sizeof(double));
    } else {
      simobj[n].solver.StageBoundaryIntegral = (double*) calloc (bf_size,sizeof(double));
      simobj[n].solver.StepBoundaryIntegral  = (double*) calloc (bf_size,sizeof(double));
    }
#else
    simobj[n].solver.StageBoundaryIntegral = (double*) calloc (bf_size,sizeof(double));
    simobj[n].solver.StepBoundaryIntegral  = (double*) calloc (bf_size,sizeof(double));
#endif

    /* initialize function call counts to zero */
    simobj[n].solver.count_hyp
      = simobj[n].solver.count_par
      = simobj[n].solver.count_sou
      = 0;

  }

  return 0;
}
