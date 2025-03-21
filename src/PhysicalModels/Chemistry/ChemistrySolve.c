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
int ChemistrySetPhotonDensity( void*   s,    /*!< Solver object of type #HyPar */
                               void*   p,    /*!< Object of type #Chemistry */
                               void*   m,    /*!< MPI object of type #MPIVariables */
                               double  a_t   /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)        s;
  MPIVariables *mpi    = (MPIVariables*) m;
  Chemistry    *params = (Chemistry*)    p;

  int *dim    = solver->dim_local;
  int ghosts  = solver->ghosts;
  int nz = params->z_i+1;

  // get xmin of the domain
  double x0 = 0.0;
  _GetCoordinate_(0,0,dim,ghosts,solver->x,x0);
  MPIMin_double(&x0, &x0, 1, &mpi->world);

  int index[solver->ndims];
  int done = 0; _ArraySetValue_(index,solver->ndims,0);
  while (!done) {
    int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

    // first z-layer
    if (a_t <= params->t_pulse_norm) {
      double x;
      _GetCoordinate_(0,index[0],dim,ghosts,solver->x,x);

      double I0 = params->I0 * ( params->IA
                                 + params->IB * cos(   params->kg * params->L_ref
                                                     * (x-x0) * (1.0 - params->IC*(x-x0)) ));
      double c = params->c;
      double h = params->h;
      double nu = params->nu;
      params->nv_hnu[nz*p+0] = I0 / (c*h*nu*params->n_O2);
    } else {
      params->nv_hnu[nz*p+0] = 0.0;
    }

    // remaining z-layers
    int iz;
    for (iz = 1; iz < nz; iz++) {
      double sigma = params->sO3 * params->n_O2;
      double damp_fac = 1.0 - params->dz*sigma*params->nv_O3old[nz*p+(iz-1)];
      params->nv_hnu[nz*p+iz] = params->nv_hnu[nz*p+(iz-1)] * damp_fac;
    }

    _ArrayIncrementIndex_(solver->ndims,dim,index,done);

  }

  return 0;
}

/*! solve the reaction equations  */
int ChemistrySolve(  void*   s,    /*!< Solver object of type #HyPar */
                     void*   p,    /*!< Object of type #Chemistry */
                     void*   m,    /*!< MPI object of type #MPIVariables */
                     double  a_dt, /*!< time step size */
                     double  a_t   /*!< Current simulation time */ )
{
  HyPar        *solver = (HyPar*)        s;
  MPIVariables *mpi    = (MPIVariables*) m;
  Chemistry    *params = (Chemistry*)    p;

  int *dim    = solver->dim_local;
  int ghosts  = solver->ghosts;

  ChemistrySetPhotonDensity( solver, params, mpi, a_t );

  // Solve the reaction equations
  int nspecies = params->nspecies;
  int iz, nz = params->z_i+1;
  for (iz = 0; iz < nz; iz++) {

    int index[solver->ndims];
    int done = 0; _ArraySetValue_(index,solver->ndims,0);
    while (!done) {
      int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

      double uchem[nspecies];
      uchem[0] = params->nv_O2[nz*p+iz];
      uchem[1] = params->nv_O3[nz*p+iz];
      uchem[2] = params->nv_1D[nz*p+iz];
      uchem[3] = params->nv_1Dg[nz*p+iz];
      uchem[4] = params->nv_3Su[nz*p+iz];
      uchem[5] = params->nv_1Sg[nz*p+iz];
      uchem[6] = params->nv_CO2[nz*p+iz];
      uchem[7] = params->nv_hnu[nz*p+iz];

      params->nv_O3old[nz*p+iz] = params->nv_O3[nz*p+iz];

      if (!strcmp(params->ti_scheme, "RK4")) {

        // RK4
        int s;

        // 1st stage
        double u1[nspecies];
        _ArrayCopy1D_(uchem, u1, nspecies);
        double f1[nspecies-2];
        _ChemistrySetRHS_(f1, u1, params);

        // 2nd stage
        double u2[nspecies];
        _ArrayCopy1D_(uchem, u2, nspecies);
        for (s = 0; s < nspecies-2; s++) u2[1+s] += 0.5*a_dt*f1[s];
        double f2[nspecies-2];
        _ChemistrySetRHS_(f2, u2, params);

        // 3rd stage
        double u3[nspecies];
        _ArrayCopy1D_(uchem, u3, nspecies);
        for (s = 0; s < nspecies-2; s++) u3[1+s] += 0.5*a_dt*f2[s];
        double f3[nspecies-2];
        _ChemistrySetRHS_(f3, u3, params);

        // 4th stage
        double u4[nspecies];
        _ArrayCopy1D_(uchem, u4, nspecies);
        for (s = 0; s < nspecies-2; s++) u4[1+s] += a_dt*f3[s];
        double f4[nspecies-2];
        _ChemistrySetRHS_(f4, u3, params);

        // final
        params->nv_O3[nz*p+iz]  += a_dt*(f1[0]+2.0*f2[0]+2.0*f3[0]+f4[0])/6.0;
        params->nv_1D[nz*p+iz]  += a_dt*(f1[1]+2.0*f2[1]+2.0*f3[1]+f4[1])/6.0;
        params->nv_1Dg[nz*p+iz] += a_dt*(f1[2]+2.0*f2[2]+2.0*f3[2]+f4[2])/6.0;
        params->nv_3Su[nz*p+iz] += a_dt*(f1[3]+2.0*f2[3]+2.0*f3[3]+f4[3])/6.0;
        params->nv_1Sg[nz*p+iz] += a_dt*(f1[4]+2.0*f2[4]+2.0*f3[4]+f4[4])/6.0;
        params->nv_CO2[nz*p+iz] += a_dt*(f1[5]+2.0*f2[5]+2.0*f3[5]+f4[5])/6.0;

      } else {

        // Forward Euler
        double fchem[nspecies-2];
        _ChemistrySetRHS_(fchem, uchem, params);

        params->nv_O3[nz*p+iz]  += a_dt * fchem[0];
        params->nv_1D[nz*p+iz]  += a_dt * fchem[1];
        params->nv_1Dg[nz*p+iz] += a_dt * fchem[2];
        params->nv_3Su[nz*p+iz] += a_dt * fchem[3];
        params->nv_1Sg[nz*p+iz] += a_dt * fchem[4];
        params->nv_CO2[nz*p+iz] += a_dt * fchem[5];

      }

      _ArrayIncrementIndex_(solver->ndims,dim,index,done);
    }

  }

  // Set the heating term
  int index[solver->ndims];
  int done = 0; _ArraySetValue_(index,solver->ndims,0);
  while (!done) {
    int p; _ArrayIndex1D_(solver->ndims,dim,index,ghosts,p);

    double uchem[nspecies];
    uchem[0] = params->nv_O2[nz*p+params->z_i];
    uchem[1] = params->nv_O3[nz*p+params->z_i];
    uchem[2] = params->nv_1D[nz*p+params->z_i];
    uchem[3] = params->nv_1Dg[nz*p+params->z_i];
    uchem[4] = params->nv_3Su[nz*p+params->z_i];
    uchem[5] = params->nv_1Sg[nz*p+params->z_i];
    uchem[6] = params->nv_CO2[nz*p+params->z_i];
    uchem[7] = params->nv_hnu[nz*p+params->z_i];

    _ChemistrySetQ_(params->Qv[p], uchem, params);

    _ArrayIncrementIndex_(solver->ndims,dim,index,done);
  }

  return 0;
}

