// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file Euler1DUpwind.c
    @author Debojyoti Ghosh
    @brief Contains functions to compute the upwind flux at grid interfaces for the 1D Euler equations.
*/

#include <stdlib.h>
#include <math.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <mathfunctions.h>
#include <matmult_native.h>
#include <physicalmodels/euler1d.h>
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
int Euler1DUpwindRoe(
                      double  *fI, /*!< Computed upwind interface flux */
                      double  *fL, /*!< Left-biased reconstructed interface flux */
                      double  *fR, /*!< Right-biased reconstructed interface flux */
                      double  *uL, /*!< Left-biased reconstructed interface solution */
                      double  *uR, /*!< Right-biased reconstructed interface solution */
                      double  *u,  /*!< Cell-centered solution */
                      int     dir, /*!< Spatial dimension (unused since this is a 1D system) */
                      void    *s,  /*!< Solver object of type #HyPar */
                      double  t    /*!< Current solution time */
                    )
{
  HyPar     *solver = (HyPar*)    s;
  Euler1D   *param  = (Euler1D*)  solver->physics;
  int       done,k;
  _DECLARE_IERR_;

  const int ndims = solver->ndims;
  const int ghosts= solver->ghosts;
  const int *dim  = solver->dim_local;
  const int nvars = param->nvars;

  int index_outer[ndims], index_inter[ndims], bounds_outer[ndims], bounds_inter[ndims];
  _ArrayCopy1D_(dim,bounds_outer,ndims); bounds_outer[dir] =  1;
  _ArrayCopy1D_(dim,bounds_inter,ndims); bounds_inter[dir] += 1;
  double R[nvars*nvars], D[nvars*nvars], L[nvars*nvars],
         DL[nvars*nvars], modA[nvars*nvars];

  done = 0; _ArraySetValue_(index_outer,ndims,0);
  while (!done) {
    _ArrayCopy1D_(index_outer,index_inter,ndims);
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D_(ndims,bounds_inter,index_inter,0,p);
      int indexL[ndims]; _ArrayCopy1D_(index_inter,indexL,ndims); indexL[dir]--;
      int indexR[ndims]; _ArrayCopy1D_(index_inter,indexR,ndims);
      int pL; _ArrayIndex1D_(ndims,dim,indexL,ghosts,pL);
      int pR; _ArrayIndex1D_(ndims,dim,indexR,ghosts,pR);
      double udiff[nvars], uavg[nvars],udiss[nvars];

      /* Roe's upwinding scheme */

      for (k = 0; k < nvars; k++) {
        udiff[k] = 0.5 * (uR[nvars*p+k] - uL[nvars*p+k]);
      }

      _Euler1DRoeAverage_         (uavg,(u+nvars*pL),(u+nvars*pR),param);
      _Euler1DEigenvalues_        (uavg,D,param,0);
      _Euler1DLeftEigenvectors_   (uavg,L,param,0);
      _Euler1DRightEigenvectors_  (uavg,R,param,0);

      for (k = 0; k < nvars; k++) {
        D[k*nvars+k] = absolute(D[k*nvars+k]);
      }

      MatMult(nvars,DL,D,L);
      MatMult(nvars,modA,R,DL);
      MatVecMult(nvars,udiss,modA,udiff);

      for (k = 0; k < nvars; k++) {
        fI[nvars*p+k] = 0.5 * (fL[nvars*p+k]+fR[nvars*p+k]) - udiss[k];
      }
    }
    _ArrayIncrementIndex_(ndims,bounds_outer,index_outer,done);
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
int Euler1DUpwindRF(
                      double  *fI, /*!< Computed upwind interface flux */
                      double  *fL, /*!< Left-biased reconstructed interface flux */
                      double  *fR, /*!< Right-biased reconstructed interface flux */
                      double  *uL, /*!< Left-biased reconstructed interface solution */
                      double  *uR, /*!< Right-biased reconstructed interface solution */
                      double  *u,  /*!< Cell-centered solution */
                      int     dir, /*!< Spatial dimension (unused since this is a 1D system) */
                      void    *s,  /*!< Solver object of type #HyPar */
                      double  t    /*!< Current solution time */
                    )
{
  HyPar     *solver = (HyPar*)    s;
  Euler1D   *param  = (Euler1D*)  solver->physics;
  int       done,k;
  _DECLARE_IERR_;

  const int ndims   = solver->ndims;
  const int *dim    = solver->dim_local;
  const int ghosts  = solver->ghosts;
  const int nvars = param->nvars;

  int index_outer[ndims], index_inter[ndims], bounds_outer[ndims], bounds_inter[ndims];
  _ArrayCopy1D_(dim,bounds_outer,ndims); bounds_outer[dir] =  1;
  _ArrayCopy1D_(dim,bounds_inter,ndims); bounds_inter[dir] += 1;
  double R[nvars*nvars], D[nvars*nvars], L[nvars*nvars];

  done = 0; _ArraySetValue_(index_outer,ndims,0);
  while (!done) {
    _ArrayCopy1D_(index_outer,index_inter,ndims);
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D_(ndims,bounds_inter,index_inter,0,p);
      int indexL[ndims]; _ArrayCopy1D_(index_inter,indexL,ndims); indexL[dir]--;
      int indexR[ndims]; _ArrayCopy1D_(index_inter,indexR,ndims);
      int pL; _ArrayIndex1D_(ndims,dim,indexL,ghosts,pL);
      int pR; _ArrayIndex1D_(ndims,dim,indexR,ghosts,pR);
      double uavg[nvars], fcL[nvars], fcR[nvars],
             ucL[nvars], ucR[nvars], fc[nvars];

      /* Roe-Fixed upwinding scheme */

      _Euler1DRoeAverage_       (uavg,(u+nvars*pL),(u+nvars*pR),param);
      _Euler1DEigenvalues_      (uavg,D,param,0);
      _Euler1DLeftEigenvectors_ (uavg,L,param,0);
      _Euler1DRightEigenvectors_(uavg,R,param,0);

      /* calculate characteristic fluxes and variables */
      MatVecMult(nvars,ucL,L,(uL+nvars*p));
      MatVecMult(nvars,ucR,L,(uR+nvars*p));
      MatVecMult(nvars,fcL,L,(fL+nvars*p));
      MatVecMult(nvars,fcR,L,(fR+nvars*p));

      for (k = 0; k < nvars; k++) {
        double eigL,eigC,eigR;
        _Euler1DEigenvalues_((u+nvars*pL),D,param,0);
        eigL = D[k*nvars+k];
        _Euler1DEigenvalues_((u+nvars*pR),D,param,0);
        eigR = D[k*nvars+k];
        _Euler1DEigenvalues_(uavg,D,param,0);
        eigC = D[k*nvars+k];

        if ((eigL > 0) && (eigC > 0) && (eigR > 0))       fc[k] = fcL[k];
        else if ((eigL < 0) && (eigC < 0) && (eigR < 0))  fc[k] = fcR[k];
        else {
          double alpha = max3(absolute(eigL),absolute(eigC),absolute(eigR));
          fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k]-ucR[k]));
        }

      }

      /* calculate the interface flux from the characteristic flux */
      MatVecMult(nvars,(fI+nvars*p),R,fc);
    }
    _ArrayIncrementIndex_(ndims,bounds_outer,index_outer,done);
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
int Euler1DUpwindLLF(
                      double  *fI, /*!< Computed upwind interface flux */
                      double  *fL, /*!< Left-biased reconstructed interface flux */
                      double  *fR, /*!< Right-biased reconstructed interface flux */
                      double  *uL, /*!< Left-biased reconstructed interface solution */
                      double  *uR, /*!< Right-biased reconstructed interface solution */
                      double  *u,  /*!< Cell-centered solution */
                      int     dir, /*!< Spatial dimension (unused since this is a 1D system) */
                      void    *s,  /*!< Solver object of type #HyPar */
                      double  t    /*!< Current solution time */
                    )
{
  HyPar     *solver = (HyPar*)    s;
  Euler1D   *param  = (Euler1D*)  solver->physics;
  int       done,k;
  _DECLARE_IERR_;

  const int ndims   = solver->ndims;
  const int *dim    = solver->dim_local;
  const int ghosts  = solver->ghosts;
  const int nvars = param->nvars;

  int index_outer[ndims], index_inter[ndims], bounds_outer[ndims], bounds_inter[ndims];
  _ArrayCopy1D_(dim,bounds_outer,ndims); bounds_outer[dir] =  1;
  _ArrayCopy1D_(dim,bounds_inter,ndims); bounds_inter[dir] += 1;
  double R[nvars*nvars], D[nvars*nvars], L[nvars*nvars];

  done = 0; _ArraySetValue_(index_outer,ndims,0);
  while (!done) {
    _ArrayCopy1D_(index_outer,index_inter,ndims);
    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {
      int p; _ArrayIndex1D_(ndims,bounds_inter,index_inter,0,p);
      int indexL[ndims]; _ArrayCopy1D_(index_inter,indexL,ndims); indexL[dir]--;
      int indexR[ndims]; _ArrayCopy1D_(index_inter,indexR,ndims);
      int pL; _ArrayIndex1D_(ndims,dim,indexL,ghosts,pL);
      int pR; _ArrayIndex1D_(ndims,dim,indexR,ghosts,pR);
      double uavg[nvars], fcL[nvars], fcR[nvars],
             ucL[nvars], ucR[nvars], fc[nvars];

      /* Local Lax-Friedrich upwinding scheme */

      _Euler1DRoeAverage_       (uavg,(u+nvars*pL),(u+nvars*pR),param);
      _Euler1DEigenvalues_      (uavg,D,param,0);
      _Euler1DLeftEigenvectors_ (uavg,L,param,0);
      _Euler1DRightEigenvectors_(uavg,R,param,0);

      /* calculate characteristic fluxes and variables */
      MatVecMult(nvars,ucL,L,(uL+nvars*p));
      MatVecMult(nvars,ucR,L,(uR+nvars*p));
      MatVecMult(nvars,fcL,L,(fL+nvars*p));
      MatVecMult(nvars,fcR,L,(fR+nvars*p));

      for (k = 0; k < nvars; k++) {
        double eigL,eigC,eigR;
        _Euler1DEigenvalues_((u+nvars*pL),D,param,0);
        eigL = D[k*nvars+k];
        _Euler1DEigenvalues_((u+nvars*pR),D,param,0);
        eigR = D[k*nvars+k];
        _Euler1DEigenvalues_(uavg,D,param,0);
        eigC = D[k*nvars+k];

        double alpha = max3(absolute(eigL),absolute(eigC),absolute(eigR));
        fc[k] = 0.5 * (fcL[k] + fcR[k] + alpha * (ucL[k]-ucR[k]));
      }

      /* calculate the interface flux from the characteristic flux */
      MatVecMult(nvars,(fI+nvars*p),R,fc);
    }
    _ArrayIncrementIndex_(ndims,bounds_outer,index_outer,done);
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
int Euler1DUpwindRusanov(
                          double  *fI, /*!< Computed upwind interface flux */
                          double  *fL, /*!< Left-biased reconstructed interface flux */
                          double  *fR, /*!< Right-biased reconstructed interface flux */
                          double  *uL, /*!< Left-biased reconstructed interface solution */
                          double  *uR, /*!< Right-biased reconstructed interface solution */
                          double  *u,  /*!< Cell-centered solution */
                          int     dir, /*!< Spatial dimension (unused since this is a 1D system) */
                          void    *s,  /*!< Solver object of type #HyPar */
                          double  t    /*!< Current solution time */
                        )
{
  HyPar     *solver = (HyPar*)    s;
  Euler1D   *param  = (Euler1D*)  solver->physics;
  int       done,k;
  _DECLARE_IERR_;

  const int ndims = solver->ndims;
  const int ghosts= solver->ghosts;
  const int *dim  = solver->dim_local;
  const int nvars = param->nvars;

  int index_outer[ndims], index_inter[ndims], bounds_outer[ndims], bounds_inter[ndims];
  _ArrayCopy1D_(dim,bounds_outer,ndims); bounds_outer[dir] =  1;
  _ArrayCopy1D_(dim,bounds_inter,ndims); bounds_inter[dir] += 1;

  double udiff[nvars], uavg[nvars];

  done = 0; _ArraySetValue_(index_outer,ndims,0);
  while (!done) {

    _ArrayCopy1D_(index_outer,index_inter,ndims);

    for (index_inter[dir] = 0; index_inter[dir] < bounds_inter[dir]; index_inter[dir]++) {

      int p; _ArrayIndex1D_(ndims,bounds_inter,index_inter,0,p);
      int indexL[ndims]; _ArrayCopy1D_(index_inter,indexL,ndims); indexL[dir]--;
      int indexR[ndims]; _ArrayCopy1D_(index_inter,indexR,ndims);
      int pL; _ArrayIndex1D_(ndims,dim,indexL,ghosts,pL);
      int pR; _ArrayIndex1D_(ndims,dim,indexR,ghosts,pR);

      _Euler1DRoeAverage_(uavg,(u+nvars*pL),(u+nvars*pR),param);
      for (k = 0; k < nvars; k++) udiff[k] = 0.5 * (uR[nvars*p+k] - uL[nvars*p+k]);

      double rho, uvel, E, P, c;

      _Euler1DGetFlowVar_((u+nvars*pL),rho,uvel,E,P,param);
      c = param->gamma*P/rho;
      double alphaL = c + absolute(uvel);

      _Euler1DGetFlowVar_((u+nvars*pR),rho,uvel,E,P,param);
      c = param->gamma*P/rho;
      double alphaR = c + absolute(uvel);

      _Euler1DGetFlowVar_(uavg,rho,uvel,E,P,param);
      c = param->gamma*P/rho;
      double alphaavg = c + absolute(uvel);

      double alpha = max3(alphaL,alphaR,alphaavg);

      for (k = 0; k < nvars; k++) {
        fI[nvars*p+k] = 0.5 * (fL[nvars*p+k]+fR[nvars*p+k]) - alpha*udiff[k];
      }

    }

    _ArrayIncrementIndex_(ndims,bounds_outer,index_outer,done);

  }

  return(0);
}
