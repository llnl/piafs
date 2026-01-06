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
    
    /* Initialize GPU optimization fields */
    solver->gpu_dim_local = NULL;
    solver->gpu_stride_with_ghosts = NULL;
    solver->gpu_stride_without_ghosts = NULL;
    solver->gpu_reduce_buffer = NULL;
    solver->gpu_reduce_buffer_size = 0;
    solver->gpu_reduce_result = NULL;
    
    /* Initialize parabolic workspace buffers */
    solver->gpu_parabolic_workspace_Q = NULL;
    solver->gpu_parabolic_workspace_QDerivX = NULL;
    solver->gpu_parabolic_workspace_QDerivY = NULL;
    solver->gpu_parabolic_workspace_QDerivZ = NULL;
    solver->gpu_parabolic_workspace_FViscous = NULL;
    solver->gpu_parabolic_workspace_FDeriv = NULL;
    solver->gpu_parabolic_workspace_size = 0;
    
#if defined(GPU_CUDA) || defined(GPU_HIP)
    solver->gpu_stream_hyp = NULL;
    solver->gpu_stream_par = NULL;
    solver->gpu_stream_sou = NULL;
#endif
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
    
    /* GPU Optimization: Cache metadata arrays on device */
#ifndef GPU_NONE
    if (GPUAllocate((void**)&solver->gpu_dim_local, solver->ndims * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate gpu_dim_local\n");
      return 1;
    }
    if (GPUCopyToDevice(solver->gpu_dim_local, solver->dim_local, solver->ndims * sizeof(int))) {
      fprintf(stderr, "Error: Failed to copy dim_local to GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->gpu_stride_with_ghosts, solver->ndims * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate gpu_stride_with_ghosts\n");
      return 1;
    }
    if (GPUCopyToDevice(solver->gpu_stride_with_ghosts, solver->stride_with_ghosts, solver->ndims * sizeof(int))) {
      fprintf(stderr, "Error: Failed to copy stride_with_ghosts to GPU\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->gpu_stride_without_ghosts, solver->ndims * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate gpu_stride_without_ghosts\n");
      return 1;
    }
    if (GPUCopyToDevice(solver->gpu_stride_without_ghosts, solver->stride_without_ghosts, solver->ndims * sizeof(int))) {
      fprintf(stderr, "Error: Failed to copy stride_without_ghosts to GPU\n");
      return 1;
    }
    
    /* GPU Optimization: Allocate persistent reduction buffers */
    /* Estimate maximum grid size for reductions (conservative: full domain with ghosts) */
    int max_points = 1;
    for (int d = 0; d < solver->ndims; d++) {
      max_points *= (solver->dim_local[d] + 2 * solver->ghosts);
    }
    max_points *= solver->nvars;
    
    /* Allocate buffers for max block count (assume block size 256) */
    int max_blocks = (max_points + 255) / 256;
    solver->gpu_reduce_buffer_size = max_blocks * sizeof(double);
    
    if (GPUAllocate((void**)&solver->gpu_reduce_buffer, solver->gpu_reduce_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_reduce_buffer\n");
      return 1;
    }
    
    if (GPUAllocate((void**)&solver->gpu_reduce_result, sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate gpu_reduce_result\n");
      return 1;
    }
    
    /* GPU Optimization: Allocate persistent parabolic workspace buffers */
    /* Size: npoints_local_wghosts * nvars */
    solver->gpu_parabolic_workspace_size = solver->npoints_local_wghosts * solver->nvars;
    size_t parabolic_buffer_size = solver->gpu_parabolic_workspace_size * sizeof(double);
    
    if (GPUAllocate((void**)&solver->gpu_parabolic_workspace_Q, parabolic_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_parabolic_workspace_Q\n");
      return 1;
    }
    if (GPUAllocate((void**)&solver->gpu_parabolic_workspace_QDerivX, parabolic_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_parabolic_workspace_QDerivX\n");
      return 1;
    }
    if (GPUAllocate((void**)&solver->gpu_parabolic_workspace_QDerivY, parabolic_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_parabolic_workspace_QDerivY\n");
      return 1;
    }
    if (GPUAllocate((void**)&solver->gpu_parabolic_workspace_QDerivZ, parabolic_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_parabolic_workspace_QDerivZ\n");
      return 1;
    }
    if (GPUAllocate((void**)&solver->gpu_parabolic_workspace_FViscous, parabolic_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_parabolic_workspace_FViscous\n");
      return 1;
    }
    if (GPUAllocate((void**)&solver->gpu_parabolic_workspace_FDeriv, parabolic_buffer_size)) {
      fprintf(stderr, "Error: Failed to allocate gpu_parabolic_workspace_FDeriv\n");
      return 1;
    }
    
    /* GPU Optimization: Create CUDA/HIP streams for overlap */
#if defined(GPU_CUDA) || defined(GPU_HIP)
    if (GPUCreateStreams(&solver->gpu_stream_hyp, &solver->gpu_stream_par, &solver->gpu_stream_sou)) {
      fprintf(stderr, "Error: Failed to create GPU streams\n");
      return 1;
    }
#endif
#endif
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
    
    /* GPU Optimization: Free cached metadata arrays */
    GPUFree(solver->gpu_dim_local);
    GPUFree(solver->gpu_stride_with_ghosts);
    GPUFree(solver->gpu_stride_without_ghosts);
    
    /* GPU Optimization: Free persistent reduction buffers */
    GPUFree(solver->gpu_reduce_buffer);
    GPUFree(solver->gpu_reduce_result);
    
    /* GPU Optimization: Free parabolic workspace buffers */
    GPUFree(solver->gpu_parabolic_workspace_Q);
    GPUFree(solver->gpu_parabolic_workspace_QDerivX);
    GPUFree(solver->gpu_parabolic_workspace_QDerivY);
    GPUFree(solver->gpu_parabolic_workspace_QDerivZ);
    GPUFree(solver->gpu_parabolic_workspace_FViscous);
    GPUFree(solver->gpu_parabolic_workspace_FDeriv);
    
    /* GPU Optimization: Destroy streams */
#if defined(GPU_CUDA) || defined(GPU_HIP)
    GPUDestroyStreams(solver->gpu_stream_hyp, solver->gpu_stream_par, solver->gpu_stream_sou);
#endif
    
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
    solver->gpu_dim_local = NULL;
    solver->gpu_stride_with_ghosts = NULL;
    solver->gpu_stride_without_ghosts = NULL;
    solver->gpu_reduce_buffer = NULL;
    solver->gpu_reduce_result = NULL;
    solver->gpu_parabolic_workspace_Q = NULL;
    solver->gpu_parabolic_workspace_QDerivX = NULL;
    solver->gpu_parabolic_workspace_QDerivY = NULL;
    solver->gpu_parabolic_workspace_QDerivZ = NULL;
    solver->gpu_parabolic_workspace_FViscous = NULL;
    solver->gpu_parabolic_workspace_FDeriv = NULL;
#if defined(GPU_CUDA) || defined(GPU_HIP)
    solver->gpu_stream_hyp = NULL;
    solver->gpu_stream_par = NULL;
    solver->gpu_stream_sou = NULL;
#endif
    /* x and dxinv (host versions) are freed in Cleanup.c */
  }
}

