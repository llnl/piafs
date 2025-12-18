/*! @file gpu_initialize_example.c
    @brief Example of GPU-enabled memory allocation for Initialize()
    
    This file shows how to modify Initialize() to allocate arrays on GPU.
    This is a reference implementation - the actual Initialize() function
    should be modified to use these patterns.
*/

#include <gpu.h>
#include <hypar.h>
#include <stdio.h>

/* Example: Allocate solution arrays on GPU */
int GPUAllocateSolutionArrays(HyPar *solver)
{
  int i;
  size_t size;
  
  /* Calculate size for solution arrays */
  size = 1;
  for (i = 0; i < solver->ndims; i++) {
    size *= (solver->dim_local[i] + 2 * solver->ghosts);
  }
  size *= solver->nvars * sizeof(double);
  
  /* Allocate on GPU */
  if (GPUAllocate((void**)&solver->u, size)) {
    fprintf(stderr, "Error: Failed to allocate u on GPU\n");
    return 1;
  }
  
  if (GPUAllocate((void**)&solver->hyp, size)) {
    fprintf(stderr, "Error: Failed to allocate hyp on GPU\n");
    return 1;
  }
  
  if (GPUAllocate((void**)&solver->par, size)) {
    fprintf(stderr, "Error: Failed to allocate par on GPU\n");
    return 1;
  }
  
  if (GPUAllocate((void**)&solver->source, size)) {
    fprintf(stderr, "Error: Failed to allocate source on GPU\n");
    return 1;
  }
  
  /* Initialize to zero */
  GPUMemset(solver->u, 0, size);
  GPUMemset(solver->hyp, 0, size);
  GPUMemset(solver->par, 0, size);
  GPUMemset(solver->source, 0, size);
  
  return 0;
}

/* Example: Copy initial solution from host to device */
int GPUCopyInitialSolution(HyPar *solver, const double *u_host)
{
  size_t size = solver->npoints_local_wghosts * solver->nvars * sizeof(double);
  return GPUCopyToDevice(solver->u, u_host, size);
}

/* Example: Copy solution from device to host for I/O */
int GPUCopySolutionForIO(HyPar *solver, double *u_host)
{
  size_t size = solver->npoints_local_wghosts * solver->nvars * sizeof(double);
  return GPUCopyToHost(u_host, solver->u, size);
}

/* Example: Free GPU arrays */
void GPUFreeSolutionArrays(HyPar *solver)
{
  GPUFree(solver->u);
  GPUFree(solver->hyp);
  GPUFree(solver->par);
  GPUFree(solver->source);
  
  solver->u = NULL;
  solver->hyp = NULL;
  solver->par = NULL;
  solver->source = NULL;
}

