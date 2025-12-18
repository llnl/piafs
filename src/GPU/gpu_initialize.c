/*! @file gpu_initialize.c
    @brief GPU-enabled memory allocation functions
*/

#include <stdio.h>
#include <stdlib.h>
#include <gpu.h>
#include <hypar.h>
#include <simulation_object.h>

/* Allocate solution arrays on GPU */
int GPUAllocateSolutionArrays(SimulationObject *simobj, int nsims)
{
  int n, i;
  
  for (n = 0; n < nsims; n++) {
    HyPar *solver = &(simobj[n].solver);
    size_t size;
    
    /* Calculate size for solution arrays */
    size = 1;
    for (i = 0; i < solver->ndims; i++) {
      size *= (solver->dim_local[i] + 2 * solver->ghosts);
    }
    size *= solver->nvars;
    
    /* Allocate main solution arrays on GPU */
    if (GPUAllocate((void**)&solver->u, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate u on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->hyp, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate hyp on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->par, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate par on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->source, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate source on GPU\n");
      return 1;
    }
    
    /* Initialize to zero */
    GPUMemset(solver->u, 0, size * sizeof(double));
    GPUMemset(solver->hyp, 0, size * sizeof(double));
    GPUMemset(solver->par, 0, size * sizeof(double));
    GPUMemset(solver->source, 0, size * sizeof(double));
    
    /* Allocate cell-centered arrays */
    if (GPUAllocate((void**)&solver->uC, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate uC on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->fluxC, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate fluxC on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->Deriv1, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate Deriv1 on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->Deriv2, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate Deriv2 on GPU\n");
      return 1;
    }
    
    /* Initialize to zero */
    GPUMemset(solver->uC, 0, size * sizeof(double));
    GPUMemset(solver->fluxC, 0, size * sizeof(double));
    GPUMemset(solver->Deriv1, 0, size * sizeof(double));
    GPUMemset(solver->Deriv2, 0, size * sizeof(double));
    
    /* Allocate node-centered arrays (interfaces) */
    size = 1;
    for (i = 0; i < solver->ndims; i++) {
      size *= (solver->dim_local[i] + 1);
    }
    size *= solver->nvars;
    solver->ndof_nodes = size;
    
    if (GPUAllocate((void**)&solver->fluxI, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate fluxI on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->uL, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate uL on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->uR, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate uR on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->fL, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate fL on GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->fR, size * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate fR on GPU\n");
      return 1;
    }
    
    /* Initialize to zero */
    GPUMemset(solver->fluxI, 0, size * sizeof(double));
    GPUMemset(solver->uL, 0, size * sizeof(double));
    GPUMemset(solver->uR, 0, size * sizeof(double));
    GPUMemset(solver->fL, 0, size * sizeof(double));
    GPUMemset(solver->fR, 0, size * sizeof(double));
  }
  
  return 0;
}

/* Allocate grid arrays on GPU */
int GPUAllocateGridArrays(SimulationObject *simobj, int nsims)
{
  int n, i;
  
  for (n = 0; n < nsims; n++) {
    HyPar *solver = &(simobj[n].solver);
    size_t size = 0;
    
    for (i = 0; i < solver->ndims; i++) {
      size += (solver->dim_local[i] + 2 * solver->ghosts);
    }
    solver->size_x = size;
    
#ifdef GPU_NONE
    /* CPU-only build: no device pointers */
    solver->d_x = NULL;
    solver->d_dxinv = NULL;
#else
    /* Allocate device copies if not already allocated (may be allocated in Initialize.c) */
    /* Host versions are allocated in Initialize.c and will be copied to device */
    if (!solver->d_x) {
      if (GPUAllocate((void**)&solver->d_x, size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate d_x on GPU\n");
        return 1;
      }
    }
    
    if (!solver->d_dxinv) {
      if (GPUAllocate((void**)&solver->d_dxinv, size * sizeof(double))) {
        fprintf(stderr, "Error: Failed to allocate d_dxinv on GPU\n");
        return 1;
      }
    }
#endif
  }
  
  return 0;
}

/* Copy grid arrays from host to device after initialization */
int GPUCopyGridArraysToDevice(SimulationObject *simobj, int nsims)
{
  int n;
  
  for (n = 0; n < nsims; n++) {
    HyPar *solver = &(simobj[n].solver);
    size_t size = solver->size_x * sizeof(double);
    
    if (!solver->x || !solver->d_x || !solver->dxinv || !solver->d_dxinv) {
      fprintf(stderr, "Error: NULL pointer detected before copy\n");
      return 1;
    }
    
    if (GPUCopyToDevice(solver->d_x, solver->x, size)) {
      fprintf(stderr, "Error: Failed to copy x to GPU\n");
      return 1;
    }
    
    if (GPUCopyToDevice(solver->d_dxinv, solver->dxinv, size)) {
      fprintf(stderr, "Error: Failed to copy dxinv to GPU\n");
      return 1;
    }
  }
  
  return 0;
}

/* Free GPU arrays */
void GPUFreeSolutionArrays(SimulationObject *simobj, int nsims)
{
  int n;
  
  for (n = 0; n < nsims; n++) {
    HyPar *solver = &(simobj[n].solver);
    
    GPUFree(solver->u);
    GPUFree(solver->hyp);
    GPUFree(solver->par);
    GPUFree(solver->source);
    GPUFree(solver->uC);
    GPUFree(solver->fluxC);
    GPUFree(solver->Deriv1);
    GPUFree(solver->Deriv2);
    GPUFree(solver->fluxI);
    GPUFree(solver->uL);
    GPUFree(solver->uR);
    GPUFree(solver->fL);
    GPUFree(solver->fR);
    GPUFree(solver->d_x);
    GPUFree(solver->d_dxinv);
    
    /* Set to NULL to avoid double-free */
    solver->u = NULL;
    solver->hyp = NULL;
    solver->par = NULL;
    solver->source = NULL;
    solver->uC = NULL;
    solver->fluxC = NULL;
    solver->Deriv1 = NULL;
    solver->Deriv2 = NULL;
    solver->fluxI = NULL;
    solver->uL = NULL;
    solver->uR = NULL;
    solver->fL = NULL;
    solver->fR = NULL;
    solver->d_x = NULL;
    solver->d_dxinv = NULL;
    /* x and dxinv (host versions) are freed in Cleanup.c */
  }
}

