// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file Euler1DWriteChem.c
    @author Debojyoti Ghosh
    @brief Function to write out the reaction-related variables
*/
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <common.h>
#include <arrayfunctions.h>
#include <io.h>
#include <mpivars.h>
#include <hypar.h>
#include <physicalmodels/euler1d.h>
#if defined(GPU_CUDA) || defined(GPU_HIP)
#include <gpu.h>
#include <gpu_runtime.h>
#include <physicalmodels/gpu_chemistry.h>
#endif

/*! Write out the reacting species data to file */
int Euler1DWriteChem(  void*   s,    /*!< Solver object of type #HyPar */
                       void*   m,    /*!< MPI object of type #MPIVariables */
                       double  a_t   /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)          s;
  MPIVariables *mpi    = (MPIVariables*)   m;
  Euler1D      *params = (Euler1D*) solver->physics;

  if (params->include_chem) {
#if defined(GPU_CUDA) || defined(GPU_HIP)
    if (GPUShouldUse()) {
      /* ChemistryWriteSpecies needs host memory - copy u and nv_hnu from device */
      int size_u = solver->npoints_local_wghosts * solver->nvars;
      double *u_host = (double*) malloc(size_u * sizeof(double));
      if (!u_host) {
        fprintf(stderr, "Error: Failed to allocate host buffer for ChemistryWriteSpecies\n");
        return 1;
      }
      GPUCopyToHost(u_host, solver->u, size_u * sizeof(double));
      GPUChemistryCopyPhotonDensityToHost(params->chem);
      GPUSync();
      ChemistryWriteSpecies(solver, u_host, params->chem, mpi, a_t);
      free(u_host);
    } else {
      ChemistryWriteSpecies(solver, solver->u, params->chem, mpi, a_t);
    }
#else
    ChemistryWriteSpecies(solver, solver->u, params->chem, mpi, a_t);
#endif
  }

  return 0;
}
