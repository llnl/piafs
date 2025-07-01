/*! @file NavierStokes2DUpwind.c
    @author Debojyoti Ghosh
    @brief Contains functions to compute the upwind flux at grid interfaces for the 2D Navier Stokes equations.
*/
#include <stdlib.h>
#include <math.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <physicalmodels/navierstokes2d.h>
#include <mathfunctions.h>
#include <matmult_native.h>
#include <hypar.h>

/*! Roe's upwinding scheme.
    \f{equation}{
      {\bf f}_{j+1/2} = \frac{1}{2}\left[ {\bf f}_{j+1/2}^L + {\bf f}_{j+1/2}^R
                         - \left| A\left({\bf u}_{j+1/2}^L,{\bf u}_{j+1/2}^R\right) \right|
                           \left( {\bf u}_{j+1/2}^R - {\bf u}_{j+1/2}^L  \right)\right]
    \f}
    + Roe, P. L., “Approximate Riemann solvers, parameter vectors, and difference schemes,” Journal of
    Computational Physics, Vol. 43, No. 2, 1981, pp. 357–372, http://dx.doi.org/10.1016/0021-9991(81)90128-5.
*/
int NavierStokes2DUpwindRoe(
                            double  *fI, /*!< Computed upwind interface flux */
                            double  *fL, /*!< Left-biased reconstructed interface flux */
                            double  *fR, /*!< Right-biased reconstructed interface flux */
                            double  *uL, /*!< Left-biased reconstructed interface solution */
                            double  *uR, /*!< Right-biased reconstructed interface solution */
                            double  *u,  /*!< Cell-centered solution */
                            int     dir, /*!< Spatial dimension (x or y) */
                            void    *s,  /*!< Solver object of type #HyPar */
                            double  t    /*!< Current solution time */
                           )
{
  HyPar           *solver = (HyPar*)          s;
  NavierStokes2D  *param  = (NavierStokes2D*) solver->physics;
  int             done, k;

  int *dim  = solver->dim_local;
  const int nvars = param->nvars;

  int bounds_outer[2], bounds_inter[2];
  bounds_outer[0] = dim[0]; bounds_outer[1] = dim[1]; bounds_outer[dir] = 1;
  bounds_inter[0] = dim[0]; bounds_inter[1] = dim[1]; bounds_inter[dir]++;
  double R[nvars*nvars], D[nvars*nvars],
         L[nvars*nvars], DL[nvars*nvars],
         modA[nvars*nvars];

  done = 0; int index_outer[2] = {0,0}; int index_inter[2];
  while (!done) {
    index_inter[0] = index_outer[0]; index_inter[1] = index_outer[1];
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D2_(_MODEL_NDIMS_,bounds_inter,index_inter,0,p);
      int indexL[_MODEL_NDIMS_]; _ArrayCopy1D_(index_inter,indexL,_MODEL_NDIMS_); indexL[dir]--;
      int indexR[_MODEL_NDIMS_]; _ArrayCopy1D_(index_inter,indexR,_MODEL_NDIMS_);
      int pL; _ArrayIndex1D_(_MODEL_NDIMS_,dim,indexL,solver->ghosts,pL);
      int pR; _ArrayIndex1D_(_MODEL_NDIMS_,dim,indexR,solver->ghosts,pR);
      double udiff[nvars], uavg[nvars],udiss[nvars];

      /* Roe's upwinding scheme */

      for (k = 0; k < nvars; k++) {
        udiff[k] = 0.5 * (uR[nvars*p+k] - uL[nvars*p+k]);
      }

      _NavierStokes2DRoeAverage_        (uavg,(u+nvars*pL),(u+nvars*pR),param->nvars,param->gamma);
      _NavierStokes2DEigenvalues_       (uavg,D,param->gamma,param->nvars,dir);
      _NavierStokes2DLeftEigenvectors_  (uavg,L,param->gamma,param->nvars,dir);
      _NavierStokes2DRightEigenvectors_ (uavg,R,param->gamma,param->nvars,dir);

       /* Harten's Entropy Fix - Page 362 of Leveque */
      double delta = 0.000001, delta2 = delta*delta;
      for (k = 0; k < nvars; k++) {
        D[k*nvars+k] = (absolute(D[k*nvars+k]) < delta ? (D[k*nvars+k]*D[k*nvars+k]+delta2)/(2*delta) : absolute(D[k*nvars+k]) );
      }

      MatMult(nvars,DL,D,L);
      MatMult(nvars,modA,R,DL);
      MatVecMult(nvars,udiss,modA,udiff);

      for (k = 0; k < nvars; k++) {
        fI[nvars*p+k] = 0.5 * (fL[nvars*p+k]+fR[nvars*p+k]) - udiss[k];
      }
    }
    _ArrayIncrementIndex_(_MODEL_NDIMS_,bounds_outer,index_outer,done);
  }

  return(0);
}

/*! Characteristic-based Roe-fixed upwinding scheme.
    \f{align}{
      \alpha_{j+1/2}^{k,L} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf f}_{j+1/2}^{k,L}, \\
      \alpha_{j+1/2}^{k,R} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf f}_{j+1/2}^{k,R}, \\
      v_{j+1/2}^{k,L} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf u}_{j+1/2}^{k,L}, \\
      v_{j+1/2}^{k,R} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf u}_{j+1/2}^{k,R}, \\
      \alpha_{j+1/2}^k &= \left\{ \begin{array}{cc} \alpha_{j+1/2}^{k,L} & {\rm if}\ \lambda_{j,j+1/2,j+1} > 0 \\ \alpha_{j+1/2}^{k,R} & {\rm if}\ \lambda_{j,j+1/2,j+1} < 0 \\ \frac{1}{2}\left[ \alpha_{j+1/2}^{k,L} + \alpha_{j+1/2}^{k,R} - \left(\max_{\left[j,j+1\right]} \lambda\right) \left( v_{j+1/2}^{k,R} - v_{j+1/2}^{k,L} \right) \right] & {\rm otherwise} \end{array}\right., \\
      {\bf f}_{j+1/2} &= \sum_{k=1}^3 \alpha_{j+1/2}^k {\bf r}_{j+1/2}^k
    \f}
    where \f${\bf l}\f$, \f${\bf r}\f$, and \f$\lambda\f$ are the left-eigenvectors, right-eigenvectors and eigenvalues. The subscripts denote the grid locations.
    + C.-W. Shu, and S. Osher, "Efficient implementation of essentially non-oscillatory schemes, II", J. Comput. Phys., 83 (1989), pp. 32–78, http://dx.doi.org/10.1016/0021-9991(89)90222-2.
*/
int NavierStokes2DUpwindRF(
                            double  *fI, /*!< Computed upwind interface flux */
                            double  *fL, /*!< Left-biased reconstructed interface flux */
                            double  *fR, /*!< Right-biased reconstructed interface flux */
                            double  *uL, /*!< Left-biased reconstructed interface solution */
                            double  *uR, /*!< Right-biased reconstructed interface solution */
                            double  *u,  /*!< Cell-centered solution */
                            int     dir, /*!< Spatial dimension (x or y) */
                            void    *s,  /*!< Solver object of type #HyPar */
                            double  t    /*!< Current solution time */
                           )
{
  HyPar    *solver = (HyPar*)    s;
  NavierStokes2D  *param  = (NavierStokes2D*)  solver->physics;
  int      done, k;

  int *dim  = solver->dim_local;
  const int nvars = param->nvars;

  int bounds_outer[2], bounds_inter[2];
  bounds_outer[0] = dim[0]; bounds_outer[1] = dim[1]; bounds_outer[dir] = 1;
  bounds_inter[0] = dim[0]; bounds_inter[1] = dim[1]; bounds_inter[dir]++;
  double R[nvars*nvars], D[nvars*nvars],
         L[nvars*nvars];

  done = 0; int index_outer[2] = {0,0}, index_inter[2];
  while (!done) {
    index_inter[0] = index_outer[0]; index_inter[1] = index_outer[1];
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D2_(_MODEL_NDIMS_,bounds_inter,index_inter,0,p);
      double uavg[nvars], fcL[nvars], fcR[nvars],
             ucL[nvars], ucR[nvars], fc[nvars];

      /* Roe-Fixed upwinding scheme */

      _NavierStokes2DRoeAverage_(uavg,(uL+nvars*p),(uR+nvars*p),param->nvars,param->gamma);
      _NavierStokes2DEigenvalues_(uavg,D,param->gamma,param->nvars,dir);
      _NavierStokes2DLeftEigenvectors_ (uavg,L,param->gamma,param->nvars,dir);
      _NavierStokes2DRightEigenvectors_(uavg,R,param->gamma,param->nvars,dir);

      /* calculate characteristic fluxes and variables */
      MatVecMult(nvars,ucL,L,(uL+nvars*p));
      MatVecMult(nvars,ucR,L,(uR+nvars*p));
      MatVecMult(nvars,fcL,L,(fL+nvars*p));
      MatVecMult(nvars,fcR,L,(fR+nvars*p));

      double eigL[nvars],eigC[nvars],eigR[nvars];
      _NavierStokes2DEigenvalues_((uL+nvars*p),D,param->gamma,param->nvars,dir);
      for (k = 0; k < nvars; k++) { eigL[k] = D[k*nvars+k]; }
      _NavierStokes2DEigenvalues_((uR+nvars*p),D,param->gamma,param->nvars,dir);
      for (k = 0; k < nvars; k++) { eigR[k] = D[k*nvars+k]; }
      _NavierStokes2DEigenvalues_(uavg,D,param->gamma,param->nvars,dir);
      for (k = 0; k < nvars; k++) { eigC[k] = D[k*nvars+k]; }

      for (k = 0; k < nvars; k++) {
        if ((eigL[k] > 0) && (eigC[k] > 0) && (eigR[k] > 0))      fc[k] = fcL[k];
        else if ((eigL[k] < 0) && (eigC[k] < 0) && (eigR[k] < 0)) fc[k] = fcR[k];
        else {
          double alpha = max3(absolute(eigL[k]),absolute(eigC[k]),absolute(eigR[k]));
          fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k]-ucR[k]));
        }
      }

      /* calculate the interface flux from the characteristic flux */
      MatVecMult(nvars,(fI+nvars*p),R,fc);
    }
    _ArrayIncrementIndex_(_MODEL_NDIMS_,bounds_outer,index_outer,done);
  }

  return(0);
}

/*! Characteristic-based local Lax-Friedrich upwinding scheme.
    \f{align}{
      \alpha_{j+1/2}^{k,L} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf f}_{j+1/2}^{k,L}, \\
      \alpha_{j+1/2}^{k,R} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf f}_{j+1/2}^{k,R}, \\
      v_{j+1/2}^{k,L} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf u}_{j+1/2}^{k,L}, \\
      v_{j+1/2}^{k,R} &= \sum_{k=1}^3 {\bf l}_{j+1/2}^k \cdot {\bf u}_{j+1/2}^{k,R}, \\
      \alpha_{j+1/2}^k &= \frac{1}{2}\left[ \alpha_{j+1/2}^{k,L} + \alpha_{j+1/2}^{k,R} - \left(\max_{\left[j,j+1\right]} \lambda\right) \left( v_{j+1/2}^{k,R} - v_{j+1/2}^{k,L} \right) \right], \\
      {\bf f}_{j+1/2} &= \sum_{k=1}^3 \alpha_{j+1/2}^k {\bf r}_{j+1/2}^k
    \f}
    where \f${\bf l}\f$, \f${\bf r}\f$, and \f$\lambda\f$ are the left-eigenvectors, right-eigenvectors and eigenvalues. The subscripts denote the grid locations.
    + C.-W. Shu, and S. Osher, "Efficient implementation of essentially non-oscillatory schemes, II", J. Comput. Phys., 83 (1989), pp. 32–78, http://dx.doi.org/10.1016/0021-9991(89)90222-2.
*/
int NavierStokes2DUpwindLLF(
                            double  *fI, /*!< Computed upwind interface flux */
                            double  *fL, /*!< Left-biased reconstructed interface flux */
                            double  *fR, /*!< Right-biased reconstructed interface flux */
                            double  *uL, /*!< Left-biased reconstructed interface solution */
                            double  *uR, /*!< Right-biased reconstructed interface solution */
                            double  *u,  /*!< Cell-centered solution */
                            int     dir, /*!< Spatial dimension (x or y) */
                            void    *s,  /*!< Solver object of type #HyPar */
                            double  t    /*!< Current solution time */
                           )
{
  HyPar    *solver = (HyPar*)    s;
  NavierStokes2D  *param  = (NavierStokes2D*)  solver->physics;
  int      done, k;

  int *dim  = solver->dim_local;
  const int nvars = param->nvars;

  int bounds_outer[2], bounds_inter[2];
  bounds_outer[0] = dim[0]; bounds_outer[1] = dim[1]; bounds_outer[dir] = 1;
  bounds_inter[0] = dim[0]; bounds_inter[1] = dim[1]; bounds_inter[dir]++;
  double R[nvars*nvars], D[nvars*nvars],
         L[nvars*nvars];

  done = 0; int index_outer[2] = {0,0}, index_inter[2];
  while (!done) {
    index_inter[0] = index_outer[0]; index_inter[1] = index_outer[1];
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D2_(_MODEL_NDIMS_,bounds_inter,index_inter,0,p);
      int indexL[_MODEL_NDIMS_]; _ArrayCopy1D_(index_inter,indexL,_MODEL_NDIMS_); indexL[dir]--;
      int indexR[_MODEL_NDIMS_]; _ArrayCopy1D_(index_inter,indexR,_MODEL_NDIMS_);
      int pL; _ArrayIndex1D_(_MODEL_NDIMS_,dim,indexL,solver->ghosts,pL);
      int pR; _ArrayIndex1D_(_MODEL_NDIMS_,dim,indexR,solver->ghosts,pR);
      double uavg[nvars], fcL[nvars], fcR[nvars],
             ucL[nvars], ucR[nvars], fc[nvars];

      /* Local Lax-Friedrich upwinding scheme */

      _NavierStokes2DRoeAverage_(uavg,(uL+nvars*p),(uR+nvars*p),param->nvars,param->gamma);
      _NavierStokes2DEigenvalues_(uavg,D,param->gamma,param->nvars,dir);
      _NavierStokes2DLeftEigenvectors_ (uavg,L,param->gamma,param->nvars,dir);
      _NavierStokes2DRightEigenvectors_(uavg,R,param->gamma,param->nvars,dir);

      /* calculate characteristic fluxes and variables */
      MatVecMult(nvars,ucL,L,(uL+nvars*p));
      MatVecMult(nvars,ucR,L,(uR+nvars*p));
      MatVecMult(nvars,fcL,L,(fL+nvars*p));
      MatVecMult(nvars,fcR,L,(fR+nvars*p));

      double eigL[nvars],eigC[nvars],eigR[nvars];
      _NavierStokes2DEigenvalues_((uL+nvars*p),D,param->gamma,param->nvars,dir);
      for (k = 0; k < nvars; k++) { eigL[k] = D[k*nvars+k]; }
      _NavierStokes2DEigenvalues_((uR+nvars*p),D,param->gamma,param->nvars,dir);
      for (k = 0; k < nvars; k++) { eigR[k] = D[k*nvars+k]; }
      _NavierStokes2DEigenvalues_(uavg,D,param->gamma,param->nvars,dir);
      for (k = 0; k < nvars; k++) { eigC[k] = D[k*nvars+k]; }

      for (k = 0; k < nvars; k++) {
        double alpha = max3(absolute(eigL[k]),absolute(eigC[k]),absolute(eigR[k]));
        fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k]-ucR[k]));
      }

      /* calculate the interface flux from the characteristic flux */
      MatVecMult(nvars,(fI+nvars*p),R,fc);
    }
    _ArrayIncrementIndex_(_MODEL_NDIMS_,bounds_outer,index_outer,done);
  }

  return(0);
}

/*! Rusanov's upwinding scheme.
    \f{equation}{
      {\bf f}_{j+1/2} = \frac{1}{2}\left[ {\bf f}_{j+1/2}^L + {\bf f}_{j+1/2}^R
                         - \max_{j,j+1} \nu_j \left( {\bf u}_{j+1/2}^R - {\bf u}_{j+1/2}^L  \right)\right]
    \f}
    where \f$\nu = c + \left|u\right|\f$.
    + Rusanov, V. V., "The calculation of the interaction of non-stationary shock waves and obstacles," USSR
    Computational Mathematics and Mathematical Physics, Vol. 1, No. 2, 1962, pp. 304–320
*/
int NavierStokes2DUpwindRusanov(
                                double  *fI, /*!< Computed upwind interface flux */
                                double  *fL, /*!< Left-biased reconstructed interface flux */
                                double  *fR, /*!< Right-biased reconstructed interface flux */
                                double  *uL, /*!< Left-biased reconstructed interface solution */
                                double  *uR, /*!< Right-biased reconstructed interface solution */
                                double  *u,  /*!< Cell-centered solution */
                                int     dir, /*!< Spatial dimension (x or y) */
                                void    *s,  /*!< Solver object of type #HyPar */
                                double  t    /*!< Current solution time */
                               )
{
  HyPar           *solver = (HyPar*)          s;
  NavierStokes2D  *param  = (NavierStokes2D*) solver->physics;
  int             *dim    = solver->dim_local, done, k;

  const int nvars = param->nvars;
  int bounds_outer[2], bounds_inter[2];
  bounds_outer[0] = dim[0]; bounds_outer[1] = dim[1]; bounds_outer[dir] = 1;
  bounds_inter[0] = dim[0]; bounds_inter[1] = dim[1]; bounds_inter[dir]++;

  done = 0; int index_outer[2] = {0,0}; int index_inter[2];
  while (!done) {
    index_inter[0] = index_outer[0]; index_inter[1] = index_outer[1];
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D2_(_MODEL_NDIMS_,bounds_inter,index_inter,0,p);
      int indexL[_MODEL_NDIMS_]; _ArrayCopy1D_(index_inter,indexL,_MODEL_NDIMS_); indexL[dir]--;
      int indexR[_MODEL_NDIMS_]; _ArrayCopy1D_(index_inter,indexR,_MODEL_NDIMS_);
      int pL; _ArrayIndex1D_(_MODEL_NDIMS_,dim,indexL,solver->ghosts,pL);
      int pR; _ArrayIndex1D_(_MODEL_NDIMS_,dim,indexR,solver->ghosts,pR);
      double udiff[nvars],uavg[nvars];

      /* Modified Rusanov's upwinding scheme */

      for (k = 0; k < nvars; k++) {
        udiff[k] = 0.5 * (uR[nvars*p+k] - uL[nvars*p+k]);
      }

      _NavierStokes2DRoeAverage_ (uavg,(u+nvars*pL),(u+nvars*pR),param->nvars,param->gamma);

      double c, vel[_MODEL_NDIMS_], rho,E,P;
      _NavierStokes2DGetFlowVar_((u+nvars*pL),rho,vel[0],vel[1],E,P,param->gamma);
      c = sqrt(param->gamma*P/rho);
      double alphaL = c + absolute(vel[dir]), betaL = absolute(vel[dir]);
      _NavierStokes2DGetFlowVar_((u+nvars*pR),rho,vel[0],vel[1],E,P,param->gamma);
      c = sqrt(param->gamma*P/rho);
      double alphaR = c + absolute(vel[dir]), betaR = absolute(vel[dir]);
      _NavierStokes2DGetFlowVar_(uavg,rho,vel[0],vel[1],E,P,param->gamma);
      c = sqrt(param->gamma*P/rho);
      double alphaavg = c + absolute(vel[dir]), betaavg = absolute(vel[dir]);

      double alpha  = max3(alphaL,alphaR,alphaavg);

      for (k = 0; k < nvars; k++) {
        fI[nvars*p+k] = 0.5*(fL[nvars*p+k]+fR[nvars*p+k])-alpha*udiff[k];
      }
    }
    _ArrayIncrementIndex_(_MODEL_NDIMS_,bounds_outer,index_outer,done);
  }

  return(0);
}
