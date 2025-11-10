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

static double hnu_first_layer(const Chemistry* const a_chem,
                              const int a_p,
                              const double a_t )
{
  double sigma = a_chem->t_pulse_norm/2.35482;
  double I0 = 0.0;
  if (a_t > a_chem->t_start_norm) {
    double tp = a_t - (a_chem->t_start_norm + a_chem->t_pulse_norm);
    I0 = a_chem->I0 * exp( -(tp*tp)/(2*sigma*sigma) ) * a_chem->imap[a_p];
  }
  double c = a_chem->c;
  double h = a_chem->h;
  double nu = a_chem->nu;
  return I0 / (c*h*nu*a_chem->n_O2);
}

static double hnu_damp_factor( const Chemistry* const a_chem,
                               const double a_nO3 )
{
  double sigma = a_chem->sO3 * a_chem->n_O2;
  double damp_fac = 1.0 - a_chem->dz*sigma*a_nO3;
  return damp_fac;
}

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

  if (solver->ndims == 3) {

    int i, j, k;
    int imax = dim[0], jmax = dim[1], kmax = dim[2];

    int my_rank_z = mpi->ip[_ZDIR_];
    int num_rank_z = mpi->iproc[_ZDIR_];
    int first_rank_z = (mpi->ip[_ZDIR_] == 0 ? 1 : 0);

    int meow[num_rank_z]; _ArraySetValue_(meow, num_rank_z, 0);

    while (!meow[num_rank_z-1]) {

      int go = (first_rank_z ? 1 : meow[my_rank_z-1]);

      if (go && (!meow[my_rank_z])) {

        for (i = 0; i < imax; i++) {
          for (j = 0; j < jmax; j++) {
            if (first_rank_z) {
              k = 0;
              int index[3] = {i, j, k};
              int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);
              chem->nv_hnu[p] = hnu_first_layer(chem, p, a_t);
            }
            int kstart = (first_rank_z ? 1 : 0);
            for (k = kstart; k < kmax; k++) {
              int p, p_km1;
              {
                int index[3] = {i, j, k-1};
                _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p_km1);
              }
              {
                int index[3] = {i, j, k};
                _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);
              }
              double n_O3 = a_U[chem->grid_stride*p_km1 + chem->n_flow_vars + iO3];
              chem->nv_hnu[p] = chem->nv_hnu[p_km1] * hnu_damp_factor(chem, n_O3);
            }
          }
        }

      }

      MPIExchangeBoundariesnD(solver->ndims, 1, dim, ghosts, mpi, chem->nv_hnu);

      meow[my_rank_z] = 1;
      MPIMax_integer(meow, meow, num_rank_z, &mpi->world);
    }

  } else {

    const int nz = chem->z_i+1;
    int done = 0;
    int index[solver->ndims]; _ArraySetValue_(index,solver->ndims,0);
    while (!done) {
      int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);
      // first z-layer
      chem->nv_hnu[nz*p+0] = hnu_first_layer(chem, p, a_t);
      // remaining z-layers
      int iz;
      for (iz = 1; iz < nz; iz++) {
        double n_O3 = a_U[chem->grid_stride*p + chem->n_flow_vars + chem->z_stride*(iz-1) + iO3];
        chem->nv_hnu[nz*p+iz] = chem->nv_hnu[nz*p+(iz-1)] * hnu_damp_factor(chem, n_O3);
      }
      _ArrayIncrementIndex_(solver->ndims,dim,index,done);
    }
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
  int nspecies = chem->nspecies;

  ChemistrySetPhotonDensity( solver, chem, mpi, a_U, a_t );

  // Solve the reaction equations
  int done = 0;  int index[solver->ndims]; _ArraySetValue_(index,solver->ndims,0);
  while (!done) {
    int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

    int iz, nz = (solver->ndims == 3 ? 1 : chem->z_i+1);
    for (iz = 0; iz < nz; iz++) {

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

    }
    // done
    _ArrayIncrementIndex_(solver->ndims,dim,index,done);
  }

  return 0;
}
