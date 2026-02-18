// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file OutputSolution.cpp
    @author Debojyoti Ghosh
    @brief Write out the solution to file
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <basic.h>
#include <common_cpp.h>
#include <arrayfunctions.h>
#include <io_cpp.h>
#include <timeintegration_cpp.h>
#include <mpivars_cpp.h>
#include <simulation_object.h>
#ifdef GPU_CUDA
#include <gpu.h>
#include <gpu_runtime.h>
#elif defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#endif

/* Function declarations */
extern "C" void IncrementFilenameIndex(char*,int);

/*! Write out the solution to file */
int OutputSolution( void*   s,      /*!< Array of simulation objects of type #SimulationObject */
                    int     nsims,  /*!< Number of simulation objects */
                    double  a_time  /*!< Current simulation time */)
{
  SimulationObject* simobj = (SimulationObject*) s;
  int ns;
  _DECLARE_IERR_;

  for (ns = 0; ns < nsims; ns++) {

    HyPar*        solver = &(simobj[ns].solver);
    MPIVariables* mpi    = &(simobj[ns].mpi);

    if (!solver->WriteOutput) continue;

    int  nu;
    char fname_root[_MAX_STRING_SIZE_];
    strcpy(fname_root, solver->op_fname_root);

    if (nsims > 1) {
      char index[_MAX_STRING_SIZE_];
      GetStringFromInteger(ns, index, (int)log10(nsims)+1);
      strcat(fname_root, "_");
      strcat(fname_root, index);
    }

    /* WriteArray accesses u and x directly - need host copies if GPU is enabled */
#ifdef GPU_CUDA
    if (GPUShouldUse()) {
      int size_u = solver->npoints_local_wghosts * solver->nvars;
      int size_x = solver->size_x;
      
      /* Safety checks */
      if (!solver->u || !solver->x) {
        fprintf(stderr, "Error: OutputSolution: solver->u=%p or solver->x=%p is NULL\n", 
                solver->u, solver->x);
        continue;
      }
      if (size_u <= 0 || size_x <= 0) {
        fprintf(stderr, "Error: OutputSolution: invalid sizes: size_u=%d, size_x=%d\n", 
                size_u, size_x);
        continue;
      }
      
      /* x is always on host, only need to copy u from device */
      double *u_host = (double*) malloc(size_u * sizeof(double));
      
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for output (size_u=%d)\n", size_u);
        continue;
      }
      
      if (GPUCopyToHost(u_host, solver->u, size_u * sizeof(double))) {
        fprintf(stderr, "Error: GPUCopyToHost failed for u\n");
        free(u_host);
        continue;
      }
      
      /* Sync to ensure copy is complete */
      GPUSync();
      
      /* Check for any GPU errors after sync - but don't exit on error during output */
      int gpu_err = GPU_GET_LAST_ERROR();
      if (gpu_err != GPU_SUCCESS) {
        fprintf(stderr, "Warning: GPU error detected after sync (error: %d), but continuing\n", gpu_err);
      }
      
      /* Verify host array is valid */
      if (!u_host) {
        fprintf(stderr, "Error: Host array is NULL after copy\n");
        free(u_host);
        continue;
      }
      
      int ierr = WriteArray(  solver->ndims,
                              solver->nvars,
                              solver->dim_global,
                              solver->dim_local,
                              solver->ghosts,
                              solver->x,
                              u_host,
                              solver,
                              mpi,
                              fname_root );
      
      
      if (ierr) {
        fprintf(stderr, "Error: WriteArray returned error code %d\n", ierr);
      }
      
      free(u_host);
      
    } else {
      WriteArray(  solver->ndims,
                   solver->nvars,
                   solver->dim_global,
                   solver->dim_local,
                   solver->ghosts,
                   solver->x,
                   solver->u,
                   solver,
                   mpi,
                   fname_root );
    }
#elif defined(GPU_HIP)
    if (GPUShouldUse()) {
      int size_u = solver->npoints_local_wghosts * solver->nvars;
      int size_x = solver->size_x;
      
      /* Safety checks */
      if (!solver->u || !solver->x) {
        fprintf(stderr, "Error: OutputSolution: solver->u=%p or solver->x=%p is NULL\n", 
                solver->u, solver->x);
        continue;
      }
      if (size_u <= 0 || size_x <= 0) {
        fprintf(stderr, "Error: OutputSolution: invalid sizes: size_u=%d, size_x=%d\n", 
                size_u, size_x);
        continue;
      }
      
      /* x is always on host, only need to copy u from device */
      double *u_host = (double*) malloc(size_u * sizeof(double));
      
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for output (size_u=%d)\n", size_u);
        continue;
      }
      
      if (GPUCopyToHost(u_host, solver->u, size_u * sizeof(double))) {
        fprintf(stderr, "Error: GPUCopyToHost failed for u\n");
        free(u_host);
        continue;
      }
      
      /* Sync to ensure copy is complete */
      GPUSync();
      
      /* Check for any GPU errors after sync - but don't exit on error during output */
      int gpu_err = GPU_GET_LAST_ERROR();
      if (gpu_err != GPU_SUCCESS) {
        fprintf(stderr, "Warning: GPU error detected after sync (error: %d), but continuing\n", gpu_err);
      }
      
      /* Verify host array is valid */
      if (!u_host) {
        fprintf(stderr, "Error: Host array is NULL after copy\n");
        free(u_host);
        continue;
      }
      
      int ierr = WriteArray(  solver->ndims,
                              solver->nvars,
                              solver->dim_global,
                              solver->dim_local,
                              solver->ghosts,
                              solver->x,
                              u_host,
                              solver,
                              mpi,
                              fname_root );
      
      
      if (ierr) {
        fprintf(stderr, "Error: WriteArray returned error code %d\n", ierr);
      }
      
      free(u_host);
      
    } else {
      WriteArray(  solver->ndims,
                   solver->nvars,
                   solver->dim_global,
                   solver->dim_local,
                   solver->ghosts,
                   solver->x,
                   solver->u,
                   solver,
                   mpi,
                   fname_root );
    }
#else
    WriteArray(  solver->ndims,
                 solver->nvars,
                 solver->dim_global,
                 solver->dim_local,
                 solver->ghosts,
                 solver->x,
                 solver->u,
                 solver,
                 mpi,
                 fname_root );
#endif

    /* increment the index string, if required */
    if ((!strcmp(solver->output_mode,"serial")) && (!strcmp(solver->op_overwrite,"no"))) {
      IncrementFilenameIndex(solver->filename_index,solver->index_length);
    }

  }

  
  /* Force a flush and small delay to see if segfault is during return */
  
  int retval = 0;
  
  return(retval);
}
