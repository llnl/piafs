/*! @file ChemistryWriteSpecies.c
    @author Debojyoti Ghosh, Albertine Oudin
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
#include <physicalmodels/chemistry.h>

/*! Write out the reacting species data to file */
int ChemistryWriteSpecies(  void*   s,    /*!< Solver object of type #HyPar */
                            void*   p,    /*!< Object of type #Chemistry */
                            void*   m,    /*!< MPI object of type #MPIVariables */
                            double  a_t   /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)        s;
  MPIVariables *mpi    = (MPIVariables*) m;
  Chemistry    *params = (Chemistry*)    p;

  int nz = params->z_i+1;
  int iz;

  int iz_start = 0;
  if (!strcmp(params->write_all_zlocs,"no")) iz_start = params->z_i;

  for (iz = iz_start; iz < nz; iz++) {

    char fname_root[_MAX_STRING_SIZE_] = "op_species";
    {
      char z_idx[_MAX_STRING_SIZE_];
      GetStringFromInteger(iz, z_idx, (int)log10(nz)+1);
      strcat(fname_root, "_z");
      strcat(fname_root, z_idx);
    }
    if (solver->nsims > 1) {
      char index[_MAX_STRING_SIZE_];
      GetStringFromInteger(solver->my_idx, index, (int)log10(solver->nsims)+1);
      strcat(fname_root, "_");
      strcat(fname_root, index);
      strcat(fname_root, "_");
    }

    int nspecies = params->nspecies;
    double* species_arr = (double*) calloc (solver->npoints_local_wghosts*nspecies, sizeof(double));

    int *dim    = solver->dim_local;
    int ghosts  = solver->ghosts;
    int index[solver->ndims], bounds[solver->ndims], offset[solver->ndims];

    /* set bounds for array index to include ghost points */
    _ArrayAddCopy1D_(dim,(2*ghosts),bounds,solver->ndims);
    /* set offset such that index is compatible with ghost point arrangement */
    _ArraySetValue_(offset,solver->ndims,-ghosts);

    int done = 0; _ArraySetValue_(index,solver->ndims,0);
    while (!done) {
      int p; _ArrayIndex1DWO_(solver->ndims,dim,index,offset,ghosts,p);
      species_arr[nspecies*p+0] = params->nv_O2[nz*p+iz];
      species_arr[nspecies*p+1] = params->nv_O3[nz*p+iz];
      species_arr[nspecies*p+2] = params->nv_1D[nz*p+iz];
      species_arr[nspecies*p+3] = params->nv_1Dg[nz*p+iz];
      species_arr[nspecies*p+4] = params->nv_3Su[nz*p+iz];
      species_arr[nspecies*p+5] = params->nv_1Sg[nz*p+iz];
      species_arr[nspecies*p+6] = params->nv_CO2[nz*p+iz];
      species_arr[nspecies*p+7] = params->nv_hnu[nz*p+iz];
      _ArrayIncrementIndex_(solver->ndims,bounds,index,done);
    }

    WriteArray(  solver->ndims,
                 nspecies,
                 solver->dim_global,
                 solver->dim_local,
                 solver->ghosts,
                 solver->x,
                 species_arr,
                 solver,
                 mpi,
                 fname_root );

    free(species_arr);
  }


  return 0;
}

