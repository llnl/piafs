// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file navierstokes2d.h
    @brief 2D Navier Stokes equations (compressible flows)
    @author Debojyoti Ghosh

  2D Navier-Stokes equations for viscous and inviscid compressible flows\n

  \f{equation}{
    \frac {\partial} {\partial t} \left[\begin{array}{c} \rho \\ \rho u \\ \rho v \\ e \end{array}\right]
  + \frac {\partial} {\partial x} \left[\begin{array}{c} \rho u \\ \rho u^2 + p \\ \rho u v \\ (e+p) u\end{array}\right]
  + \frac {\partial} {\partial y} \left[\begin{array}{c} \rho v \\ \rho u v \\ \rho v^2 + p \\ (e+p) v \end{array}\right]
  = \frac {\partial} {\partial x} \left[\begin{array}{c} 0 \\ \tau_{xx} \\ \tau_{yx} \\ u \tau_{xx} + v \tau_{yx} - q_x \end{array}\right]
  + \frac {\partial} {\partial y} \left[\begin{array}{c} 0 \\ \tau_{xy} \\ \tau_{yy} \\ u \tau_{xy} + v \tau_{yy} - q_y \end{array}\right]
  + \left[\begin{array}{c} 0 \\ 0 \\ 0  \\ \frac{Q}{\gamma-1} \end{array}\right]
  \f}
  where \f$Q\f$ is the chemical heating term, and the viscous terms are given by
  \f{align}{
    \tau_{ij} &= \frac{\mu}{Re_\infty} \left[ \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right) - \frac{2}{3}\frac{\partial u_k}{\partial x_k} \delta_{ij} \right], \\
    q_i &= - \frac{\mu}{\left(\gamma-1\right)Re_\infty Pr} \frac{\partial T}{\partial x_i}
  \f}
  with \f$\mu\f$ being the viscosity coefficient (computed using Sutherland's law), and the equation of state is
  \f{equation}{
    e = \frac {p} {\gamma-1} + \frac{1}{2} \rho \left(u^2 + v^2\right)
  \f}
  References for the governing equations (as well as non-dimensional form):-
  + Tannehill, Anderson and Pletcher, Computational Fluid Mechanics and Heat Transfer,
    Chapter 5, Section 5.1.7 (However, the non-dimensional velocity and the Reynolds
    number is based on speed of sound, instead of the freestream velocity).
*/
#include <basic.h>
#include <math_ops.h>
#include <physicalmodels/chemistry.h>

/*! 2D Navier Stokes equations */
#define _NAVIER_STOKES_2D_  "navierstokes2d"

/* define ndims and nvars for this model */
#undef _MODEL_NDIMS_
#undef _NS2D_NVARS_
/*! Number of spatial dimensions */
#define _MODEL_NDIMS_ 2
/*! Number of Navier-Stokes variables per grid point */
#define _NS2D_NVARS_ 4

/* choices for upwinding schemes */
/*! Roe's upwinding scheme */
#define _ROE_       "roe"
/*! Characteristic-based Roe-fixed scheme */
#define _RF_        "rf-char"
/*! Characteristic-based local Lax-Friedrich scheme */
#define _LLF_       "llf-char"
/*! Rusanov's upwinding scheme */
#define _RUSANOV_   "rusanov"

/* directions */
/*! dimension corresponding to the \a x spatial dimension */
#define _XDIR_ 0
/*! dimension corresponding to the \a y spatial dimension */
#define _YDIR_ 1

/*! \def _NavierStokes2DGetFlowVar_
 Get the flow variables from the conserved solution vector.
 \f{equation}{
   {\bf u} = \left[\begin{array}{c} \rho \\ \rho u \\ \rho v \\ e \\ \vdots \\ \phi_i \\ \vdots \end{array}\right]
 \f}
 where \f$\phi_i\f$ are passively-advected scalars
*/
#define _NavierStokes2DGetFlowVar_(u,rho,vx,vy,e,P,gamma) \
  { \
    double  vsq; \
    rho = u[0]; \
    vx  = (rho==0) ? 0 : u[1] / rho; \
    vy  = (rho==0) ? 0 : u[2] / rho; \
    e   = u[3]; \
    vsq  = (vx*vx) + (vy*vy); \
    P   = (e - 0.5*rho*vsq) * (gamma-1.0); \
  }

/*! \def _NavierStokes2DSetFlux_
  Compute the flux vector, given the flow variables
  \f{eqnarray}{
    dir = x, & {\bf f}\left({\bf u}\right) = \left[\begin{array}{c} \rho u \\ \rho u^2 + p \\ \rho u v \\ (e+p)u \\ \vdots \\ u \phi_i \\ \vdots \end{array}\right], \\
    dir = y, & {\bf f}\left({\bf u}\right) = \left[\begin{array}{c} \rho v \\ \rho u v \\ \rho v^2 + p \\ (e+p)v \\ \vdots \\ v \phi_i \\ \vdots \end{array}\right]
  \f}
*/
#define _NavierStokes2DSetFlux_(f,u,gamma,nvars,dir) \
  { \
    double rho, vx, vy, e, P; \
    _NavierStokes2DGetFlowVar_(u,rho,vx,vy,e,P,gamma); \
    int m_i;\
    if (dir == _XDIR_) { \
      f[0] = rho * vx; \
      f[1] = rho * vx * vx + P; \
      f[2] = rho * vx * vy; \
      f[3] = (e + P) * vx; \
      for (m_i = _NS2D_NVARS_; m_i < nvars; m_i++) { \
          f[m_i] = vx * u[m_i]; \
      } \
    } else if (dir == _YDIR_) { \
      f[0] = rho * vy; \
      f[1] = rho * vy * vx; \
      f[2] = rho * vy * vy + P; \
      f[3] = (e + P) * vy; \
      for (m_i = _NS2D_NVARS_; m_i < nvars; m_i++) { \
          f[m_i] = vy * u[m_i]; \
      } \
    } \
  }

/*! \def _NavierStokes2DRoeAverage_
  Compute the Roe-average of two solutions.
*/
#define _NavierStokes2DRoeAverage_(uavg,uL,uR,nvars,gamma) \
  { \
    double  rho ,vx, vy, e ,P ,H ,csq, vsq; \
    double  rhoL,vxL,vyL,eL,PL,HL,cLsq,vsqL; \
    double  rhoR,vxR,vyR,eR,PR,HR,cRsq,vsqR; \
    rhoL = uL[0]; \
    vxL  = uL[1] / rhoL; \
    vyL  = uL[2] / rhoL; \
    eL   = uL[3]; \
    vsqL = (vxL*vxL) + (vyL*vyL); \
    PL   = (eL - 0.5*rhoL*vsqL) * (gamma-1.0); \
    cLsq = gamma * PL/rhoL; \
    HL = 0.5*(vxL*vxL+vyL*vyL) + cLsq / (gamma-1.0); \
    rhoR = uR[0]; \
    vxR  = uR[1] / rhoR; \
    vyR  = uR[2] / rhoR; \
    eR   = uR[3]; \
    vsqR = (vxR*vxR) + (vyR*vyR); \
    PR   = (eR - 0.5*rhoR*vsqR) * (gamma-1.0); \
    cRsq = gamma * PR/rhoR; \
    HR = 0.5*(vxR*vxR+vyR*vyR) + cRsq / (gamma-1.0); \
    double tL = sqrt(rhoL); \
    double tR = sqrt(rhoR); \
    rho = tL * tR; \
    vx  = (tL*vxL + tR*vxR) / (tL + tR); \
    vy  = (tL*vyL + tR*vyR) / (tL + tR); \
    H   = (tL*HL + tR*HR) / (tL + tR); \
    vsq = vx*vx + vy*vy; \
    csq = (gamma-1.0) * (H-0.5*vsq); \
    P   = csq * rho / gamma; \
    e   = P/(gamma-1.0) + 0.5*rho*vsq; \
    uavg[0] = rho; \
    uavg[1] = rho*vx; \
    uavg[2] = rho*vy; \
    uavg[3] = e; \
    int m_i; \
    for (m_i = _NS2D_NVARS_; m_i < nvars; m_i++) { \
        uavg[m_i] = sqrt(uL[m_i]) * sqrt(uR[m_i]); \
    } \
  }

/*! \def _NavierStokes2DEigenvalues_
  Compute the eigenvalues, given a solution vector in terms of the conserved variables. The eigenvalues are returned
  as a matrix D whose diagonal values are the eigenvalues. Admittedly, this is inefficient. The matrix D is stored in
  a row-major format.
*/
#define _NavierStokes2DEigenvalues_(u,D,gamma,nvars,dir) \
  { \
    double  rho,vx,vy,e,P,c,vn,vsq; \
    _NavierStokes2DGetFlowVar_(u,rho,vx,vy,e,P,gamma); \
    vsq = (vx*vx) + (vy*vy); \
    c = sqrt(gamma*P/rho); \
    if      (dir == _XDIR_) vn = vx; \
    else if (dir == _YDIR_) vn = vy; \
    else                    vn = 0.0; \
    _ArraySetValue_(D, nvars*nvars, 0.0); \
    D[0*nvars+0] = vn-c; \
    if (dir == _XDIR_) {\
      D[1*nvars+1] = vn+c; \
      D[2*nvars+2] = vn;\
    } else if (dir == _YDIR_) { \
      D[1*nvars+1] = vn; \
      D[2*nvars+2] = vn+c; \
    }\
    D[3*nvars+3] = vn; \
    int m_i; \
    for (m_i = _NS2D_NVARS_; m_i < nvars; m_i++) { \
      D[m_i*nvars+m_i] = vn; \
    } \
  }

/*! \def _NavierStokes2DLeftEigenvectors_
  Compute the left eigenvectors, given a solution vector in terms of the conserved variables. The eigenvectors are
  returned as a matrix L whose rows correspond to each eigenvector. The matrix L is stored in the row-major format.
  \n\n
  Reference:
  + Rohde, "Eigenvalues and eigenvectors of the Euler equations in general geometries", AIAA Paper 2001-2609,
    http://dx.doi.org/10.2514/6.2001-2609
*/
#define _NavierStokes2DLeftEigenvectors_(u,L,ga,nvars,dir) \
  { \
    double ga_minus_one=ga-1.0; \
    double rho,vx,vy,e,P,a,un,ek,vsq; \
    _NavierStokes2DGetFlowVar_(u,rho,vx,vy,e,P,ga); \
    double nx = 0,ny = 0; \
    vsq  = (vx*vx) + (vy*vy); \
    ek = 0.5 * (vx*vx + vy*vy); \
    a = sqrt(ga * P / rho); \
    _ArraySetValue_(L, nvars*nvars, 0.0); \
    if (dir == _XDIR_) { \
      un = vx; \
      nx = 1.0; \
      L[0*nvars+0] = (ga_minus_one*ek + a*un) / (2*a*a); \
      L[0*nvars+1] = ((-ga_minus_one)*vx - a*nx) / (2*a*a); \
      L[0*nvars+2] = ((-ga_minus_one)*vy - a*ny) / (2*a*a); \
      L[0*nvars+3] = ga_minus_one / (2*a*a); \
      L[3*nvars+0] = (a*a - ga_minus_one*ek) / (a*a); \
      L[3*nvars+1] = (ga_minus_one*vx) / (a*a); \
      L[3*nvars+2] = (ga_minus_one*vy) / (a*a); \
      L[3*nvars+3] = (-ga_minus_one) / (a*a); \
      L[1*nvars+0] = (ga_minus_one*ek - a*un) / (2*a*a); \
      L[1*nvars+1] = ((-ga_minus_one)*vx + a*nx) / (2*a*a); \
      L[1*nvars+2] = ((-ga_minus_one)*vy + a*ny) / (2*a*a); \
      L[1*nvars+3] = ga_minus_one / (2*a*a); \
      L[2*nvars+0] = (vy - un*ny) / nx; \
      L[2*nvars+1] = ny; \
      L[2*nvars+2] = (ny*ny - 1.0) / nx; \
      L[2*nvars+3] = 0.0; \
    } else if (dir == _YDIR_) {  \
      un = vy;  \
      ny = 1.0; \
      L[0*nvars+0] = (ga_minus_one*ek+a*un) / (2*a*a); \
      L[0*nvars+1] = ((1.0-ga)*vx - a*nx) / (2*a*a); \
      L[0*nvars+2] = ((1.0-ga)*vy - a*ny) / (2*a*a); \
      L[0*nvars+3] = ga_minus_one / (2*a*a); \
      L[3*nvars+0] = (a*a-ga_minus_one*ek) / (a*a); \
      L[3*nvars+1] = ga_minus_one*vx / (a*a); \
      L[3*nvars+2] = ga_minus_one*vy / (a*a); \
      L[3*nvars+3] = (1.0 - ga) / (a*a); \
      L[2*nvars+0] = (ga_minus_one*ek-a*un) / (2*a*a); \
      L[2*nvars+1] = ((1.0-ga)*vx + a*nx) / (2*a*a); \
      L[2*nvars+2] = ((1.0-ga)*vy + a*ny) / (2*a*a); \
      L[2*nvars+3] = ga_minus_one / (2*a*a); \
      L[1*nvars+0] = (un*nx-vx) / ny; \
      L[1*nvars+1] = (1.0 - nx*nx) / ny; \
      L[1*nvars+2] = - nx; \
      L[1*nvars+3] = 0; \
    } \
    int m_i; \
    for (m_i = _NS2D_NVARS_; m_i < nvars; m_i++) { \
      L[m_i*nvars+m_i] = 1.0; \
    } \
  }

/*! \def _NavierStokes2DRightEigenvectors_
  Compute the right eigenvectors, given a solution vector in terms of the conserved variables. The eigenvectors are
  returned as a matrix R whose columns correspond to each eigenvector. The matrix R is stored in the row-major format.
  \n\n
  Reference:
  + Rohde, "Eigenvalues and eigenvectors of the Euler equations in general geometries", AIAA Paper 2001-2609,
    http://dx.doi.org/10.2514/6.2001-2609
*/
#define _NavierStokes2DRightEigenvectors_(u,R,ga,nvars,dir) \
  { \
    double ga_minus_one = ga-1.0; \
    double rho,vx,vy,e,P,un,ek,a,h0,vsq; \
    _NavierStokes2DGetFlowVar_(u,rho,vx,vy,e,P,ga); \
    double nx = 0,ny = 0; \
    vsq  = (vx*vx) + (vy*vy); \
    ek   = 0.5 * (vx*vx + vy*vy); \
    a    = sqrt(ga * P / rho); \
    h0   = a*a / ga_minus_one + ek; \
    _ArraySetValue_(R, nvars*nvars, 0.0); \
    if (dir == _XDIR_) { \
      un = vx; \
      nx = 1.0; \
      R[0*nvars+0] = 1.0; \
      R[1*nvars+0] = vx - a*nx; \
      R[2*nvars+0] = vy - a*ny; \
      R[3*nvars+0] = h0 - a*un; \
      R[0*nvars+3] = 1.0; \
      R[1*nvars+3] = vx; \
      R[2*nvars+3] = vy; \
      R[3*nvars+3] = ek; \
      R[0*nvars+1] = 1.0; \
      R[1*nvars+1] = vx + a*nx; \
      R[2*nvars+1] = vy + a*ny; \
      R[3*nvars+1] = h0 + a*un; \
      R[0*nvars+2] = 0.0; \
      R[1*nvars+2] = ny; \
      R[2*nvars+2] = -nx; \
      R[3*nvars+2] = vx*ny - vy*nx; \
    } else if (dir == _YDIR_) { \
      un = vy; \
      ny = 1.0; \
      R[0*nvars+0] = 1.0; \
      R[1*nvars+0] = vx - a*nx; \
      R[2*nvars+0] = vy - a*ny; \
      R[3*nvars+0] = h0 - a*un; \
      R[0*nvars+3] = 1.0; \
      R[1*nvars+3] = vx; \
      R[2*nvars+3] = vy; \
      R[3*nvars+3] = ek; \
      R[0*nvars+2] = 1.0; \
      R[1*nvars+2] = vx + a*nx; \
      R[2*nvars+2] = vy + a*ny; \
      R[3*nvars+2] = h0 + a*un; \
      R[0*nvars+1] = 0; \
      R[1*nvars+1] = ny; \
      R[2*nvars+1] = -nx; \
      R[3*nvars+1] = vx*ny-vy*nx; \
    } \
    int m_i; \
    for (m_i = _NS2D_NVARS_; m_i < nvars; m_i++) { \
      R[m_i*nvars+m_i] = 1.0; \
    } \
  }

/*! \def _NavierStokes2DCoeffViscosity_
    Compute the viscosity coefficient given the temperature */
#define _NavierStokes2DCoeffViscosity_(mu, T_norm, param) \
  { \
    double T_d = T_norm*param->Tref;  \
    mu = raiseto(T_d/param->T0, 1.5)  \
         * (param->T0 + param->TS)  \
         / (T_d       + param->TS); \
  }

/*! \def _NavierStokes2DCoeffConductivity_
    Compute the conductivity coefficient given the temperature */
#define _NavierStokes2DCoeffConductivity_(kappa, T_norm, param) \
  { \
    double T_d = T_norm*param->Tref; \
    kappa = raiseto(T_d/param->T0, 1.5)                                   \
            * (param->T0 + param->TA * exp(-param->TB/param->T0))       \
            / (T_d       + param->TA * exp(-param->TB/T_d      )); \
  }

/*! \def NavierStokes2D
    \brief Structure containing variables and parameters specific to the 2D Navier Stokes equations.
 *  This structure contains the physical parameters, variables, and function pointers specific to
 *  the 2D Navier-Stokes equations.
*/
/*! \brief Structure containing variables and parameters specific to the 2D Navier Stokes equations.
 *  This structure contains the physical parameters, variables, and function pointers specific to
 *  the 2D Navier-Stokes equations.
*/
typedef struct navierstokes2d_parameters {
  double  gamma;                          /*!< Ratio of heat capacities */
  char    upw_choice[_MAX_STRING_SIZE_];  /*!< choice of upwinding */
  double  Re;                             /*!< Reynolds number */
  double  Pr;                             /*!< Prandtl  number */

  int include_chem; /*!< Flag to include chemistry */
  void* chem; /*!< Photochemical reactions object */

  int nvars; /*!< Number of variables per grid point */
  char write_op[_MAX_STRING_SIZE_]; /*!< Write physics-specific output to file */

  // constants for computing viscosity and conductivity coefficients
  double Tref; /*!< Reference temperature */
  double T0; /*!< T_0 (in Kelvins) (viscoscity/conductivity coeff) */
  double TS; /*!< T_S (in Kelvins) (viscoscity/conductivity coeff) */
  double TA; /*!< T_A (in Kelvins) (viscoscity/conductivity coeff) */
  double TB; /*!< T_A (in Kelvins) (viscoscity/conductivity coeff) */

} NavierStokes2D;

int    NavierStokes2DInitialize (void*,void*);
int    NavierStokes2DCleanup    (void*);

