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
int ChemistrySetPhotonDensity( void*   a_s,    /*!< Solver object of type #HyPar */
                               void*   a_p,    /*!< Object of type #Chemistry */
                               void*   a_m,    /*!< MPI object of type #MPIVariables */
                               double  a_t   /*!< Current simulation time */ )
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
    double I0 = chem->I0 * exp( - (a_t - chem->t_pulse_norm)*(a_t - chem->t_pulse_norm) / 2 /sigma / sigma ) * chem->imap[p];
    //double I0 = chem->I0 * chem->imap[p];
    double c = chem->c;
    double h = chem->h;
    double nu = chem->nu;
    chem->nv_hnu[nz*p+0] = I0 / (c*h*nu*chem->n_O2);

    // remaining z-layers
    int iz;
    for (iz = 1; iz < nz; iz++) {
      double sigma = chem->sO3 * chem->n_O2;
      double damp_fac = 1.0 - chem->dz*sigma*chem->nv_O3old[nz*p+(iz-1)];
      chem->nv_hnu[nz*p+iz] = chem->nv_hnu[nz*p+(iz-1)] * damp_fac;
    }

    _ArrayIncrementIndex_(solver->ndims,dim,index,done);

  }

  return 0;
}

/*! solve the reaction equations  */
int ChemistrySolve(  void*   a_s,    /*!< Solver object of type #HyPar */
                     double* a_U,    /*!< Solution array */
                     void*   a_p,    /*!< Object of type #Chemistry */
                     void*   a_m,    /*!< MPI object of type #MPIVariables */
                     double  a_dt, /*!< time step size */
                     double  a_t   /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)        a_s;
  MPIVariables *mpi    = (MPIVariables*) a_m;
  Chemistry    *chem   = (Chemistry*)    a_p;

  int *dim    = solver->dim_local;
  int ghosts  = solver->ghosts;

  ChemistrySetPhotonDensity( solver, chem, mpi, a_t );

  // Solve the reaction equations
  int nspecies = chem->nspecies;
  int iz, nz = chem->z_i+1;
  for (iz = 0; iz < nz; iz++) {

    int index[solver->ndims];
    int done = 0; _ArraySetValue_(index,solver->ndims,0);
    while (!done) {
      int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

      double uchem[nspecies];
      _ArrayCopy1D_( (a_U + chem->grid_stride*p + chem->n_flow_vars + chem->z_stride*iz), uchem, chem->z_stride);
      uchem[chem->nspecies-1] = chem->nv_hnu[nz*p+iz];
      chem->nv_O3old[nz*p+iz] = uchem[1];

      if (!strcmp(chem->ti_scheme, "RK4")) {

        // RK4
        int s;

        // 1st stage
        double u1[nspecies];
        _ArrayCopy1D_(uchem, u1, nspecies);
        double f1[nspecies-2];
        _ChemistrySetRHS_(f1, u1, chem);

        // 2nd stage
        double u2[nspecies];
        _ArrayCopy1D_(uchem, u2, nspecies);
        for (s = 0; s < nspecies-2; s++) u2[1+s] += 0.5*a_dt*f1[s];
        double f2[nspecies-2];
        _ChemistrySetRHS_(f2, u2, chem);

        // 3rd stage
        double u3[nspecies];
        _ArrayCopy1D_(uchem, u3, nspecies);
        for (s = 0; s < nspecies-2; s++) u3[1+s] += 0.5*a_dt*f2[s];
        double f3[nspecies-2];
        _ChemistrySetRHS_(f3, u3, chem);

        // 4th stage
        double u4[nspecies];
        _ArrayCopy1D_(uchem, u4, nspecies);
        for (s = 0; s < nspecies-2; s++) u4[1+s] += a_dt*f3[s];
        double f4[nspecies-2];
        _ChemistrySetRHS_(f4, u3, chem);

        // final
        uchem[1] += a_dt*(f1[0]+2.0*f2[0]+2.0*f3[0]+f4[0])/6.0;
        uchem[2] += a_dt*(f1[1]+2.0*f2[1]+2.0*f3[1]+f4[1])/6.0;
        uchem[3] += a_dt*(f1[2]+2.0*f2[2]+2.0*f3[2]+f4[2])/6.0;
        uchem[4] += a_dt*(f1[3]+2.0*f2[3]+2.0*f3[3]+f4[3])/6.0;
        uchem[5] += a_dt*(f1[4]+2.0*f2[4]+2.0*f3[4]+f4[4])/6.0;
        uchem[6] += a_dt*(f1[5]+2.0*f2[5]+2.0*f3[5]+f4[5])/6.0;

      } else {

        // Forward Euler
        double fchem[nspecies-2];
        _ChemistrySetRHS_(fchem, uchem, chem);

        uchem[1] += a_dt * fchem[0];
        uchem[2] += a_dt * fchem[1];
        uchem[3] += a_dt * fchem[2];
        uchem[4] += a_dt * fchem[3];
        uchem[5] += a_dt * fchem[4];
        uchem[6] += a_dt * fchem[5];

      }

      _ArrayCopy1D_( uchem, (a_U + chem->grid_stride*p + chem->n_flow_vars + chem->z_stride*iz), chem->z_stride);
      _ArrayIncrementIndex_(solver->ndims,dim,index,done);
    }

  }

  // Set the heating term
  int index[solver->ndims];
  int done = 0; _ArraySetValue_(index,solver->ndims,0);
  while (!done) {
    int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

    double uchem[nspecies];
    _ArrayCopy1D_( (a_U + chem->grid_stride*p + chem->n_flow_vars + chem->z_stride*chem->z_i), uchem, chem->z_stride);
    uchem[nspecies-1] = chem->nv_hnu[nz*p+chem->z_i];

    _ChemistrySetQ_(chem->Qv[p], uchem, chem);
    _ArrayIncrementIndex_(solver->ndims,dim,index,done);
  }

  return 0;
}

