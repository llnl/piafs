/*! @file gpu_chemistry.c
    @brief GPU-enabled chemistry functions
    @author Debojyoti Ghosh, Albertine Oudin
*/

#include <stdlib.h>
#include <stdio.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <hypar.h>
#include <mpivars.h>
#include <physicalmodels/chemistry.h>
#include <physicalmodels/gpu_chemistry.h>

/* Forward declaration of ChemistrySetPhotonDensity */
int ChemistrySetPhotonDensity(void*, void*, void*, double*, double);

/* GPU pointers for chemistry arrays */
static double *nv_hnu_gpu = NULL;
static double *nv_hnu_host = NULL;  // Host backup of nv_hnu for ChemistrySetPhotonDensity
static double *u_host = NULL;
static int nv_hnu_size = 0;
static int u_host_size = 0;

/*! Allocate GPU memory for chemistry arrays */
int GPUChemistryAllocate(void* a_p, int npoints_total, int nz)
{
  Chemistry *chem = (Chemistry*) a_p;

  nv_hnu_size = npoints_total * nz;
  
  // Keep the original CPU pointer (already allocated in ChemistryInitialize)
  // DO NOT replace chem->nv_hnu - keep it as host pointer for CPU operations
  nv_hnu_host = chem->nv_hnu;
  
  // Allocate GPU memory for nv_hnu
  if (GPUAllocate((void**)&nv_hnu_gpu, nv_hnu_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to allocate GPU memory for nv_hnu\n");
    return 1;
  }
  
  // Initialize GPU memory to zero
  if (GPUMemset(nv_hnu_gpu, 0, nv_hnu_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to initialize GPU memory for nv_hnu\n");
    GPUFree(nv_hnu_gpu);
    return 1;
  }
  
  // Allocate host memory for U array (for ChemistrySetPhotonDensity)
  u_host_size = chem->grid_stride * npoints_total;
  u_host = (double*) malloc(u_host_size * sizeof(double));
  if (!u_host) {
    fprintf(stderr, "Error: Failed to allocate host memory for U array\n");
    GPUFree(nv_hnu_gpu);
    return 1;
  }
  
  // Keep chem->nv_hnu pointing to host memory - do NOT replace with GPU pointer
  // The GPU pointer will be used internally in GPUChemistrySource
  
  return 0;
}

/*! Free GPU memory for chemistry arrays */
int GPUChemistryFree(void* a_p)
{
  Chemistry *chem = (Chemistry*) a_p;
  
  // Free GPU memory
  if (nv_hnu_gpu) {
    GPUFree(nv_hnu_gpu);
    nv_hnu_gpu = NULL;
  }
  
  // Free host U array
  if (u_host) {
    free(u_host);
    u_host = NULL;
  }
  
  // nv_hnu_host is just a pointer backup - don't free it!
  // The actual memory is owned by chem->nv_hnu and will be freed in ChemistryCleanup
  nv_hnu_host = NULL;
  
  // Keep chem->nv_hnu pointing to host memory (don't set to NULL)
  
  nv_hnu_size = 0;
  u_host_size = 0;
  
  return 0;
}

/*! GPU-enabled chemistry source function */
int GPUChemistrySource(
  void*   a_s,  /*!< Solver object of type #HyPar */
  double* a_U,  /*!< Solution array (on GPU) */
  double* a_S,  /*!< Source array (on GPU) */
  void*   a_p,  /*!< Object of type #Chemistry */
  void*   a_m,  /*!< MPI object of type #MPIVariables */
  double  a_t   /*!< Current simulation time */
)
{
  HyPar        *solver = (HyPar*)        a_s;
  MPIVariables *mpi    = (MPIVariables*) a_m;
  Chemistry    *chem   = (Chemistry*)    a_p;

  int npoints = solver->npoints_local_wghosts;

  /* Validate that GPU memory was properly allocated */
  if (!nv_hnu_gpu || !u_host || nv_hnu_size <= 0 || u_host_size <= 0) {
    fprintf(stderr, "Error: GPUChemistrySource called but GPU memory not allocated!\n");
    fprintf(stderr, "  nv_hnu_gpu=%p, u_host=%p, nv_hnu_size=%d, u_host_size=%d\n",
            (void*)nv_hnu_gpu, (void*)u_host, nv_hnu_size, u_host_size);
    fprintf(stderr, "  Make sure GPUChemistryAllocate was called during initialization.\n");
    return 1;
  }

  // Step 1: Copy U from GPU to host for ChemistrySetPhotonDensity
  if (GPUCopyToHost(u_host, a_U, u_host_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to copy U from GPU to host\n");
    return 1;
  }
  
  // Step 2: Compute photon density on CPU (requires MPI communication)
  // chem->nv_hnu already points to host memory (nv_hnu_host), so just call directly
  ChemistrySetPhotonDensity(solver, chem, mpi, u_host, a_t);
  
  // Step 3: Copy nv_hnu from host to GPU
  // nv_hnu_host and chem->nv_hnu point to the same memory
  if (GPUCopyToDevice(nv_hnu_gpu, chem->nv_hnu, nv_hnu_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to copy nv_hnu from host to GPU\n");
    return 1;
  }
  
  // Step 4: Launch GPU kernel to compute chemistry sources
  double gamma_m1_inv = 1.0 / (chem->gamma - 1.0);
  
  gpu_launch_chemistry_source(
    a_S, a_U, nv_hnu_gpu,
    npoints, solver->nvars, chem->n_flow_vars,
    chem->grid_stride, chem->z_stride, chem->z_i,
    solver->ndims,
    chem->k0a_norm, chem->k0b_norm, chem->k1a_norm, chem->k1b_norm,
    chem->k2a_norm, chem->k2b_norm, chem->k3a_norm, chem->k3b_norm,
    chem->k4_norm, chem->k5_norm, chem->k6_norm,
    chem->q0a_norm, chem->q0b_norm, chem->q1a_norm, chem->q1b_norm,
    chem->q2a_norm, chem->q2b_norm, chem->q3a_norm, chem->q3b_norm,
    chem->q4_norm, chem->q5_norm, chem->q6_norm,
    gamma_m1_inv,
    256  // block size
  );
  
  fflush(stderr);
  
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  fflush(stderr);
  
  return 0;
}

