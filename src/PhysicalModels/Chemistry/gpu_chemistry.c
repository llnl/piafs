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
static double *imap_gpu = NULL;
static int nv_hnu_size = 0;
static int imap_size = 0;

/*! Allocate GPU memory for chemistry arrays */
int GPUChemistryAllocate(void* a_p, int npoints_total, int nz)
{
  Chemistry *chem = (Chemistry*) a_p;

  nv_hnu_size = npoints_total * nz;
  imap_size = npoints_total;

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

  // Allocate GPU memory for imap and copy from host
  if (GPUAllocate((void**)&imap_gpu, imap_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to allocate GPU memory for imap\n");
    GPUFree(nv_hnu_gpu);
    return 1;
  }

  if (GPUCopyToDevice(imap_gpu, chem->imap, imap_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to copy imap to GPU\n");
    GPUFree(nv_hnu_gpu);
    GPUFree(imap_gpu);
    return 1;
  }

  // Keep chem->nv_hnu pointing to host memory (already allocated in ChemistryInitialize)
  // The GPU pointers (nv_hnu_gpu, imap_gpu) will be used internally in GPU functions

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

  if (imap_gpu) {
    GPUFree(imap_gpu);
    imap_gpu = NULL;
  }

  // Keep chem->nv_hnu pointing to host memory (don't set to NULL)
  // The actual memory will be freed in ChemistryCleanup

  nv_hnu_size = 0;
  imap_size = 0;

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
  if (!nv_hnu_gpu || nv_hnu_size <= 0) {
    fprintf(stderr, "Error: GPUChemistrySource called but GPU memory not allocated!\n");
    fprintf(stderr, "  nv_hnu_gpu=%p, nv_hnu_size=%d\n",
            (void*)nv_hnu_gpu, nv_hnu_size);
    fprintf(stderr, "  Make sure GPUChemistryAllocate was called during initialization.\n");
    return 1;
  }

  // Step 1: Compute photon density on GPU
  if (GPUChemistrySetPhotonDensity(solver, chem, mpi, a_U, a_t)) {
    fprintf(stderr, "Error: GPUChemistrySetPhotonDensity failed\n");
    return 1;
  }

  // Step 2: Launch GPU kernel to compute chemistry sources
  double gamma_m1_inv = 1.0 / (chem->gamma - 1.0);

  gpu_launch_chemistry_source(
    a_S, a_U, nv_hnu_gpu,
    npoints, solver->nvars, chem->n_flow_vars,
    chem->grid_stride, chem->z_stride, chem->z_i,
    solver->ndims,
    solver->dim_local[0], solver->dim_local[1], solver->dim_local[2],
    solver->ghosts,
    chem->k0a_norm, chem->k0b_norm, chem->k1a_norm, chem->k1b_norm,
    chem->k2a_norm, chem->k2b_norm, chem->k3a_norm, chem->k3b_norm,
    chem->k4_norm, chem->k5_norm, chem->k6_norm,
    chem->q0a_norm, chem->q0b_norm, chem->q1a_norm, chem->q1b_norm,
    chem->q2a_norm, chem->q2b_norm, chem->q3a_norm, chem->q3b_norm,
    chem->q4_norm, chem->q5_norm, chem->q6_norm,
    gamma_m1_inv,
    256  // block size
  );

  if (GPUShouldSyncEveryOp()) GPUSync();

  return 0;
}

/*! GPU-enabled set photon density function */
int GPUChemistrySetPhotonDensity(
  void*   a_s,  /*!< Solver object of type #HyPar */
  void*   a_p,  /*!< Object of type #Chemistry */
  void*   a_m,  /*!< MPI object of type #MPIVariables */
  double* a_U,  /*!< Solution array (on GPU) */
  double  a_t   /*!< Current simulation time */
)
{
  HyPar        *solver = (HyPar*)        a_s;
  MPIVariables *mpi    = (MPIVariables*) a_m;
  Chemistry    *chem   = (Chemistry*)    a_p;

  int *dim    = solver->dim_local;
  int ghosts  = solver->ghosts;
  int npoints = solver->npoints_local_wghosts;

  /* Validate that GPU memory was properly allocated */
  if (!nv_hnu_gpu || !imap_gpu || nv_hnu_size <= 0 || imap_size <= 0) {
    fprintf(stderr, "Error: GPUChemistrySetPhotonDensity called but GPU memory not allocated!\n");
    fprintf(stderr, "  nv_hnu_gpu=%p, imap_gpu=%p, nv_hnu_size=%d, imap_size=%d\n",
            (void*)nv_hnu_gpu, (void*)imap_gpu, nv_hnu_size, imap_size);
    return 1;
  }

  // imap_gpu is already on GPU (allocated and copied in GPUChemistryAllocate)

  if (solver->ndims == 3) {
    // 3D case: Use batched kernel for all z-layers in a single launch

    int imax = dim[0], jmax = dim[1], kmax = dim[2];
    int my_rank_z = mpi->ip[_ZDIR_];
    int num_rank_z = mpi->iproc[_ZDIR_];
    int first_rank_z = (mpi->ip[_ZDIR_] == 0 ? 1 : 0);

    if (num_rank_z == 1) {
      // Single rank in z: Use optimized batched kernel (single launch for all layers)
      int kstart = (first_rank_z ? 1 : 0);
      gpu_launch_chemistry_photon_density_3d_batched(
        nv_hnu_gpu, a_U, imap_gpu,
        imax, jmax, kmax,
        dim[0], dim[1], dim[2], ghosts,
        chem->grid_stride, chem->n_flow_vars,
        chem->I0, chem->c, chem->h, chem->nu, chem->n_O2,
        chem->t_pulse_norm, chem->t_start_norm,
        chem->sO3, chem->dz, a_t,
        first_rank_z, kstart,
        16  // block size for 2D grid
      );
      /* No GPUSync needed - next kernel on same stream waits automatically */
    } else {
      // Multiple ranks in z: Need MPI synchronization between ranks
      // Use batched kernel within each rank's local domain

      int *meow = (int*) calloc(num_rank_z, sizeof(int));
      if (!meow) {
        fprintf(stderr, "Error: Failed to allocate meow array\n");
        return 1;
      }

      // Allocate host memory for nv_hnu boundary exchange
      double *nv_hnu_host_temp = (double*) malloc(npoints * sizeof(double));
      if (!nv_hnu_host_temp) {
        fprintf(stderr, "Error: Failed to allocate host temp memory\n");
        free(meow);
        return 1;
      }

      while (!meow[num_rank_z-1]) {

        int go = (first_rank_z ? 1 : meow[my_rank_z-1]);

        if (go && (!meow[my_rank_z])) {
          // Use batched kernel for this rank's z-layers (single launch)
          int kstart = (first_rank_z ? 1 : 0);
          gpu_launch_chemistry_photon_density_3d_batched(
            nv_hnu_gpu, a_U, imap_gpu,
            imax, jmax, kmax,
            dim[0], dim[1], dim[2], ghosts,
            chem->grid_stride, chem->n_flow_vars,
            chem->I0, chem->c, chem->h, chem->nu, chem->n_O2,
            chem->t_pulse_norm, chem->t_start_norm,
            chem->sO3, chem->dz, a_t,
            first_rank_z, kstart,
            16  // block size for 2D grid
          );
          /* No GPUSync needed - GPUCopyToHost uses synchronous memcpy which waits */

          meow[my_rank_z] = 1;
        }

        // Copy nv_hnu from GPU to host for MPI exchange
        // Note: GPUCopyToHost uses synchronous cudaMemcpy which waits for prior kernels
        if (GPUCopyToHost(nv_hnu_host_temp, nv_hnu_gpu, npoints * sizeof(double))) {
          fprintf(stderr, "Error: Failed to copy nv_hnu from GPU to host\n");
          free(nv_hnu_host_temp);
          free(meow);
          return 1;
        }
        /* No GPUSync needed - synchronous memcpy already completed */

        // MPI boundary exchange on host
        MPIExchangeBoundariesnD(solver->ndims, 1, dim, ghosts, mpi, nv_hnu_host_temp);

        // Copy back to GPU after exchange
        if (GPUCopyToDevice(nv_hnu_gpu, nv_hnu_host_temp, npoints * sizeof(double))) {
          fprintf(stderr, "Error: Failed to copy nv_hnu from host to GPU\n");
          free(nv_hnu_host_temp);
          free(meow);
          return 1;
        }

        // Synchronize meow array across all ranks
        MPIMax_integer(meow, meow, num_rank_z, &mpi->world);
      }

      free(nv_hnu_host_temp);
      free(meow);
    }

  } else {
    // 1D/2D case: Each grid point processes its z-stack independently

    int nz = chem->z_i + 1;

    gpu_launch_chemistry_photon_density_1d2d(
      nv_hnu_gpu, a_U, imap_gpu,
      npoints, chem->grid_stride, chem->z_stride, chem->n_flow_vars, nz,
      chem->I0, chem->c, chem->h, chem->nu, chem->n_O2,
      chem->t_pulse_norm, chem->t_start_norm, chem->sO3, chem->dz,
      a_t,
      256  // block size
    );
    /* No GPUSync needed - next kernel on same stream waits automatically */
  }

  // nv_hnu stays on GPU - no need to copy back to host
  // It will be used directly by gpu_launch_chemistry_source
  // Use GPUChemistryCopyPhotonDensityToHost() when host access is needed

  return 0;
}

/*! Copy nv_hnu from GPU to host for output/diagnostics */
int GPUChemistryCopyPhotonDensityToHost(void* a_p)
{
  Chemistry *chem = (Chemistry*) a_p;

  if (!nv_hnu_gpu || nv_hnu_size <= 0) {
    // GPU not enabled or not allocated
    return 0;
  }

  if (GPUCopyToHost(chem->nv_hnu, nv_hnu_gpu, nv_hnu_size * sizeof(double))) {
    fprintf(stderr, "Error: Failed to copy nv_hnu from GPU to host\n");
    return 1;
  }

  return 0;
}

