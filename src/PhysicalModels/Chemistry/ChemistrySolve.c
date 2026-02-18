// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file ChemistrySolve.c
    @author Debojyoti Ghosh, Albertine Oudin
    @brief Solve the photo-chemical reactions
*/

#include <stdlib.h>
#include <basic.h>
#include <common.h>
#include <arrayfunctions.h>
#include <mpivars.h>
#include <hypar.h>
#include <physicalmodels/chemistry.h>

/*! set the photon density  */
int ChemistrySetPhotonDensity( void*   a_s, /*!< Solver object of type #HyPar */
                               void*   a_p, /*!< Object of type #Chemistry */
                               void*   a_m, /*!< MPI object of type #MPIVariables */
                               double* a_U,  /*!< Solution array */
                               double  a_t  /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)        a_s;
  MPIVariables *mpi    = (MPIVariables*) a_m;
  Chemistry    *chem = (Chemistry*)    a_p;

  int *dim    = solver->dim_local;
  int ghosts  = solver->ghosts;
  int nz = chem->z_i+1;

  int index[solver->ndims];
  int done = 0; _ArraySetValue_(index,solver->ndims,0);
  while (!done) {
    int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

    // first z-layer
    double x;
    _GetCoordinate_(0,index[0],dim,ghosts,solver->x,x);
    double sigma = chem->t_pulse_norm/2.35482;
    double I0 = 0.0;
    if (a_t > chem->t_start_norm) {
      double tp = a_t - (chem->t_start_norm + chem->t_pulse_norm);
      I0 = chem->I0 * exp( -(tp*tp)/(2*sigma*sigma) ) * chem->imap[p];
    }
    //double I0 = chem->I0 * chem->imap[p];
    double c = chem->c;
    double h = chem->h;
    double nu = chem->nu;
    chem->nv_hnu[nz*p+0] = I0 / (c*h*nu*chem->n_O2);

    // remaining z-layers
    int iz;
    for (iz = 1; iz < nz; iz++) {
      double sigma = chem->sO3 * chem->n_O2;
      double damp_fac = 1.0 - chem->dz*sigma*a_U[chem->grid_stride*p + chem->n_flow_vars + chem->z_stride*(iz-1) + iO3];
      chem->nv_hnu[nz*p+iz] = chem->nv_hnu[nz*p+(iz-1)] * damp_fac;
    }

    _ArrayIncrementIndex_(solver->ndims,dim,index,done);

  }

  return 0;
}

/*! set the reaction source terms  */
int ChemistrySource( void*   a_s,  /*!< Solver object of type #HyPar */
                     double* a_U,  /*!< Solution array */
                     double* a_S,  /*!< Source array */
                     void*   a_p,  /*!< Object of type #Chemistry */
                     void*   a_m,  /*!< MPI object of type #MPIVariables */
                     double  a_t   /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)        a_s;
  MPIVariables *mpi    = (MPIVariables*) a_m;
  Chemistry    *chem   = (Chemistry*)    a_p;

  int *dim    = solver->dim_local;
  int ghosts  = solver->ghosts;

  ChemistrySetPhotonDensity( solver, chem, mpi, a_U, a_t );

  // Solve the reaction equations
  int nspecies = chem->nspecies;
  int iz, nz = chem->z_i+1;
  for (iz = 0; iz < nz; iz++) {

    int index[solver->ndims];
    int done = 0; _ArraySetValue_(index,solver->ndims,0);
    while (!done) {
      int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

      // Set reaction sources
      _ChemistrySetRHS_(  (a_S + chem->grid_stride*p),
                          (a_U + chem->grid_stride*p),
                          chem,
                          chem->nv_hnu[nz*p+iz],
                          iz );

      // Set heating source
      _ChemistrySetQ_( (*(a_S + chem->grid_stride*p + chem->n_flow_vars-1)),
                       (a_U + chem->grid_stride*p),
                       chem,
                       chem->nv_hnu[nz*p+iz],
                       iz );

      // done
      _ArrayIncrementIndex_(solver->ndims,dim,index,done);
    }

  }

  return 0;
}
