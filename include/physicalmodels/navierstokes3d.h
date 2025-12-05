/*! @file navierstokes3d.h
    @brief 3D Navier Stokes equations (compressible flows)
    @author Debojyoti Ghosh

  3D Navier-Stokes equations for viscous and inviscid compressible flows\n

  \f{equation}{
    \frac {\partial} {\partial t} \left[\begin{array}{c} \rho \\ \rho u \\ \rho v \\ \rho w \\ e \end{array}\right]
  + \frac {\partial} {\partial x} \left[\begin{array}{c} \rho u \\ \rho u^2 + p \\ \rho u v \\ \rho u w \\ (e+p) u\end{array}\right]
  + \frac {\partial} {\partial y} \left[\begin{array}{c} \rho v \\ \rho u v \\ \rho v^2 + p \\ \rho v w \\ (e+p) v \end{array}\right]
  + \frac {\partial} {\partial z} \left[\begin{array}{c} \rho w \\ \rho u w \\ \rho v w \\ \rho w^2 + p \\ (e+p) w \end{array}\right]
  = \frac {\partial} {\partial x} \left[\begin{array}{c} 0 \\ \tau_{xx} \\ \tau_{yx} \\ \tau_{zx} \\ u \tau_{xx} + v \tau_{yx} + w \tau_{zx} - q_x \end{array}\right]
  + \frac {\partial} {\partial y} \left[\begin{array}{c} 0 \\ \tau_{xy} \\ \tau_{yy} \\ \tau_{zy} \\ u \tau_{xy} + v \tau_{yy} + w \tau_{zy} - q_y \end{array}\right]
  + \frac {\partial} {\partial z} \left[\begin{array}{c} 0 \\ \tau_{xz} \\ \tau_{yz} \\ \tau_{zz} \\ u \tau_{xz} + v \tau_{yz} + w \tau_{zz} - q_z \end{array}\right]
  + \left[\begin{array}{c} 0 \\ 0 \\ 0 \\ 0 \\ \frac{Q}{\gamma-1} \end{array}\right]
  \f}
  where \f$Q\f$ is the chemical heating term, and the viscous terms are given by
  \f{align}{
    \tau_{ij} &= \frac{\mu}{Re_\infty} \left[ \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right) - \frac{2}{3}\frac{\partial u_k}{\partial x_k} \delta_{ij} \right], \\
    q_i &= - \frac{\mu}{\left(\gamma-1\right)Re_\infty Pr} \frac{\partial T}{\partial x_i}
  \f}
  with \f$\mu\f$ being the viscosity coefficient (computed using Sutherland's law), and the equation of state is
  \f{equation}{
    e = \frac {p} {\gamma-1} + \frac{1}{2} \rho \left(u^2 + v^2 + w^2\right)
  \f}
  References for the governing equations (as well as non-dimensional form):-
  + Tannehill, Anderson and Pletcher, Computational Fluid Mechanics and Heat Transfer,
    Chapter 5, Section 5.1.7 (However, the non-dimensional velocity and the Reynolds
    number is based on speed of sound, instead of the freestream velocity).
*/
#include <stdio.h>
#include <stdlib.h>
#include <basic.h>
#include <math_ops.h>
#include <physicalmodels/chemistry.h>

/*! 3D Navier Stokes equations */
#define _NAVIER_STOKES_3D_  "navierstokes3d"

/* define ndims and nvars for this model */
#undef _MODEL_NDIMS_
#undef _NS3D_NVARS_
/*! Number of spatial dimensions */
#define _MODEL_NDIMS_ 3
/*! Number of Navier-Stokes variables per grid point (rho, rho*u, rho*v, rho*w, e) */
#define _NS3D_NVARS_ 5

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
/*! dimension corresponding to the \a z spatial dimension */
#define _ZDIR_ 2

/*! \def _NavierStokes3DGetFlowVar_
 Get the flow variables from the conserved solution vector.
 \f{equation}{
   {\bf u} = \left[\begin{array}{c} \rho \\ \rho u \\ \rho v \\ \rho w \\ e \\ \vdots \\ \phi_i \\ \vdots \end{array}\right]
 \f}
 where \f$\phi_i\f$ are passively-advected scalars
*/
#define _NavierStokes3DGetFlowVar_(u,rho,vx,vy,vz,e,P,gamma) \
  { \
    double  vsq; \
    rho = u[0]; \
    if (isnan(rho) || isinf(rho)) { \
      fprintf(stderr,"ERROR in _NavierStokes3DGetFlowVar_: NaN/Inf density detected.\n"); \
      exit(1); \
    } \
    vx  = (rho==0) ? 0 : u[1] / rho; \
    vy  = (rho==0) ? 0 : u[2] / rho; \
    vz  = (rho==0) ? 0 : u[3] / rho; \
    e   = u[4]; \
    vsq  = (vx*vx) + (vy*vy) + (vz*vz); \
    P   = (e - 0.5*rho*vsq) * (gamma-1.0); \
    if (isnan(vx) || isinf(vx) || isnan(vy) || isinf(vy) || isnan(vz) || isinf(vz) || isnan(P) || isinf(P)) { \
      fprintf(stderr,"ERROR in _NavierStokes3DGetFlowVar_: NaN/Inf in velocity (%e,%e,%e) or pressure (%e).\n",vx,vy,vz,P); \
      exit(1); \
    } \
  }

/*! \def _NavierStokes3DSetFlux_
  Compute the flux vector, given the flow variables
  \f{eqnarray}{
    dir = x, & {\bf f}\left({\bf u}\right) = \left[\begin{array}{c} \rho u \\ \rho u^2 + p \\ \rho u v \\ \rho u w \\ (e+p)u \\ \vdots \\ u \phi_i \\ \vdots \end{array}\right], \\
    dir = y, & {\bf f}\left({\bf u}\right) = \left[\begin{array}{c} \rho v \\ \rho u v \\ \rho v^2 + p \\ \rho v w \\ (e+p)v \\ \vdots \\ v \phi_i \\ \vdots \end{array}\right], \\
    dir = z, & {\bf f}\left({\bf u}\right) = \left[\begin{array}{c} \rho w \\ \rho u w \\ \rho v w \\ \rho w^2 + p \\ (e+p)w \\ \vdots \\ w \phi_i \\ \vdots \end{array}\right]
  \f}
*/
#define _NavierStokes3DSetFlux_(f,u,gamma,nvars,dir) \
  { \
    double rho, vx, vy, vz, e, P; \
    _NavierStokes3DGetFlowVar_(u,rho,vx,vy,vz,e,P,gamma); \
    int m_i;\
    if (dir == _XDIR_) { \
      f[0] = rho * vx; \
      f[1] = rho * vx * vx + P; \
      f[2] = rho * vx * vy; \
      f[3] = rho * vx * vz; \
      f[4] = (e + P) * vx; \
      for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
          f[m_i] = vx * u[m_i]; \
      } \
    } else if (dir == _YDIR_) { \
      f[0] = rho * vy; \
      f[1] = rho * vy * vx; \
      f[2] = rho * vy * vy + P; \
      f[3] = rho * vy * vz; \
      f[4] = (e + P) * vy; \
      for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
          f[m_i] = vy * u[m_i]; \
      } \
    } else if (dir == _ZDIR_) { \
      f[0] = rho * vz; \
      f[1] = rho * vz * vx; \
      f[2] = rho * vz * vy; \
      f[3] = rho * vz * vz + P; \
      f[4] = (e + P) * vz; \
      for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
          f[m_i] = vz * u[m_i]; \
      } \
    } \
  }

/*! \def _NavierStokes3DRoeAverage_
  Compute the Roe-average of two solutions.
*/
#define _NavierStokes3DRoeAverage_(uavg,uL,uR,nvars,gamma) \
  { \
    double  rho ,vx, vy, vz, e ,P ,H , vsq; \
    double  rhoL,vxL,vyL,vzL,eL,PL,HL,cLsq,vsqL; \
    _NavierStokes3DGetFlowVar_(uL,rhoL,vxL,vyL,vzL,eL,PL,gamma); \
    double  rhoR,vxR,vyR,vzR,eR,PR,HR,cRsq,vsqR; \
    _NavierStokes3DGetFlowVar_(uR,rhoR,vxR,vyR,vzR,eR,PR,gamma); \
    cLsq = gamma * PL/rhoL; \
    HL = 0.5*(vxL*vxL+vyL*vyL+vzL*vzL) + cLsq / (gamma-1.0); \
    cRsq = gamma * PR/rhoR; \
    HR = 0.5*(vxR*vxR+vyR*vyR+vzR*vzR) + cRsq / (gamma-1.0); \
    double tL = sqrt(rhoL); \
    double tR = sqrt(rhoR); \
    if (isnan(tL) || isinf(tL) || isnan(tR) || isinf(tR)) { \
      fprintf(stderr,"ERROR in _NavierStokes3DRoeAverage_: NaN/Inf in sqrt(rho) detected.\n"); \
      exit(1); \
    } \
    rho = tL * tR; \
    vx  = (tL*vxL + tR*vxR) / (tL + tR); \
    vy  = (tL*vyL + tR*vyR) / (tL + tR); \
    vz  = (tL*vzL + tR*vzR) / (tL + tR); \
    H   = (tL*HL + tR*HR) / (tL + tR); \
    vsq = vx*vx + vy*vy + vz*vz; \
    P   = (gamma-1.0) * (H-0.5*vsq) * rho / gamma; \
    e   = P/(gamma-1.0) + 0.5*rho*vsq; \
    if (isnan(vx) || isinf(vx) || isnan(vy) || isinf(vy) || isnan(vz) || isinf(vz) || isnan(P) || isinf(P)) { \
      fprintf(stderr,"ERROR in _NavierStokes3DRoeAverage_: NaN/Inf in averaged quantities detected.\n"); \
      exit(1); \
    } \
    uavg[0] = rho; \
    uavg[1] = rho*vx; \
    uavg[2] = rho*vy; \
    uavg[3] = rho*vz; \
    uavg[4] = e; \
    int m_i; \
    for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
        uavg[m_i] = sqrt(uL[m_i]) * sqrt(uR[m_i]); \
    } \
  }

/*! \def _NavierStokes3DEigenvalues_
  Compute the eigenvalues, given a solution vector in terms of the conserved variables. The eigenvalues are returned
  as a matrix D whose diagonal values are the eigenvalues. Admittedly, this is inefficient. The matrix D is stored in
  a row-major format.
*/
#define _NavierStokes3DEigenvalues_(u,D,gamma,nvars,dir) \
  { \
    _ArraySetValue_(D,nvars*nvars,0.0); \
    double  rho,vx,vy,vz,e,P,c,vn; \
    _NavierStokes3DGetFlowVar_(u,rho,vx,vy,vz,e,P,gamma); \
    c = sqrt(gamma*P/rho); \
    if      (dir == _XDIR_) vn = vx; \
    else if (dir == _YDIR_) vn = vy; \
    else if (dir == _ZDIR_) vn = vz; \
    else                    vn = 0.0; \
    _ArraySetValue_(D, nvars*nvars, 0.0); \
    D[0*nvars+0] = vn; \
    if (dir == _XDIR_) {\
      D[1*nvars+1] = vn-c; \
      D[2*nvars+2] = vn;\
      D[3*nvars+3] = vn;\
    } else if (dir == _YDIR_) { \
      D[1*nvars+1] = vn; \
      D[2*nvars+2] = vn-c; \
      D[3*nvars+3] = vn;\
    } else if (dir == _ZDIR_) { \
      D[1*nvars+1] = vn; \
      D[2*nvars+2] = vn;\
      D[3*nvars+3] = vn-c; \
    }\
    D[4*nvars+4] = vn+c; \
    int m_i; \
    for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
      D[m_i*nvars+m_i] = vn; \
    } \
  }

/*! \def _NavierStokes3DLeftEigenvectors_
  Compute the left eigenvectors, given a solution vector in terms of the conserved variables. The eigenvectors are
  returned as a matrix L whose rows correspond to each eigenvector. The matrix L is stored in the row-major format.
  \n\n
  Reference:
  + Rohde, "Eigenvalues and eigenvectors of the Euler equations in general geometries", AIAA Paper 2001-2609,
    http://dx.doi.org/10.2514/6.2001-2609
*/
#define _NavierStokes3DLeftEigenvectors_(u,L,ga,nvars,dir) \
  { \
    _ArraySetValue_(L,nvars*nvars,0.0); \
    double  ga_minus_one=ga-1.0; \
    double  rho,vx,vy,vz,e,P,a,ek; \
    _NavierStokes3DGetFlowVar_(u,rho,vx,vy,vz,e,P,ga); \
    ek = 0.5 * (vx*vx + vy*vy + vz*vz); \
    a = sqrt(ga * P / rho); \
    if (dir == _XDIR_) { \
      L[1*nvars+0] = (ga_minus_one*ek + a*vx) / (2*a*a); \
      L[1*nvars+1] = ((-ga_minus_one)*vx-a) / (2*a*a); \
      L[1*nvars+2] = ((-ga_minus_one)*vy) / (2*a*a); \
      L[1*nvars+3] = ((-ga_minus_one)*vz) / (2*a*a); \
      L[1*nvars+4] = ga_minus_one / (2*a*a); \
      L[0*nvars+0] = (a*a - ga_minus_one*ek) / (a*a); \
      L[0*nvars+1] = (ga_minus_one*vx) / (a*a); \
      L[0*nvars+2] = (ga_minus_one*vy) / (a*a); \
      L[0*nvars+3] = (ga_minus_one*vz) / (a*a); \
      L[0*nvars+4] = (-ga_minus_one) / (a*a); \
      L[4*nvars+0] = (ga_minus_one*ek - a*vx) / (2*a*a); \
      L[4*nvars+1] = ((-ga_minus_one)*vx+a) / (2*a*a); \
      L[4*nvars+2] = ((-ga_minus_one)*vy) / (2*a*a); \
      L[4*nvars+3] = ((-ga_minus_one)*vz) / (2*a*a); \
      L[4*nvars+4] = ga_minus_one / (2*a*a); \
      L[2*nvars+0] = vy; \
      L[2*nvars+1] = 0.0; \
      L[2*nvars+2] = -1.0; \
      L[2*nvars+3] = 0.0; \
      L[2*nvars+4] = 0.0; \
      L[3*nvars+0] = -vz; \
      L[3*nvars+1] = 0.0; \
      L[3*nvars+2] = 0.0; \
      L[3*nvars+3] = 1.0; \
      L[3*nvars+4] = 0.0; \
    } else if (dir == _YDIR_) {  \
      L[2*nvars+0] = (ga_minus_one*ek+a*vy) / (2*a*a); \
      L[2*nvars+1] = ((1.0-ga)*vx) / (2*a*a); \
      L[2*nvars+2] = ((1.0-ga)*vy-a) / (2*a*a); \
      L[2*nvars+3] = ((1.0-ga)*vz) / (2*a*a); \
      L[2*nvars+4] = ga_minus_one / (2*a*a); \
      L[0*nvars+0] = (a*a-ga_minus_one*ek) / (a*a); \
      L[0*nvars+1] = ga_minus_one*vx / (a*a); \
      L[0*nvars+2] = ga_minus_one*vy / (a*a); \
      L[0*nvars+3] = ga_minus_one*vz / (a*a); \
      L[0*nvars+4] = (1.0 - ga) / (a*a); \
      L[4*nvars+0] = (ga_minus_one*ek-a*vy) / (2*a*a); \
      L[4*nvars+1] = ((1.0-ga)*vx) / (2*a*a); \
      L[4*nvars+2] = ((1.0-ga)*vy+a) / (2*a*a); \
      L[4*nvars+3] = ((1.0-ga)*vz) / (2*a*a); \
      L[4*nvars+4] = ga_minus_one / (2*a*a); \
      L[1*nvars+0] = -vx; \
      L[1*nvars+1] = 1.0; \
      L[1*nvars+2] = 0.0; \
      L[1*nvars+3] = 0.0; \
      L[1*nvars+4] = 0; \
      L[3*nvars+0] = vz; \
      L[3*nvars+1] = 0.0; \
      L[3*nvars+2] = 0.0; \
      L[3*nvars+3] = -1.0; \
      L[3*nvars+4] = 0; \
    } else if (dir == _ZDIR_) {  \
      L[3*nvars+0] = (ga_minus_one*ek+a*vz) / (2*a*a); \
      L[3*nvars+1] = ((1.0-ga)*vx) / (2*a*a); \
      L[3*nvars+2] = ((1.0-ga)*vy) / (2*a*a); \
      L[3*nvars+3] = ((1.0-ga)*vz-a) / (2*a*a); \
      L[3*nvars+4] = ga_minus_one / (2*a*a); \
      L[0*nvars+0] = (a*a-ga_minus_one*ek) / (a*a); \
      L[0*nvars+1] = ga_minus_one*vx / (a*a); \
      L[0*nvars+2] = ga_minus_one*vy / (a*a); \
      L[0*nvars+3] = ga_minus_one*vz / (a*a); \
      L[0*nvars+4] = (1.0-ga) / (a*a); \
      L[4*nvars+0] = (ga_minus_one*ek-a*vz) / (2*a*a); \
      L[4*nvars+1] = ((1.0-ga)*vx) / (2*a*a); \
      L[4*nvars+2] = ((1.0-ga)*vy) / (2*a*a); \
      L[4*nvars+3] = ((1.0-ga)*vz+a) / (2*a*a); \
      L[4*nvars+4] = ga_minus_one / (2*a*a); \
      L[1*nvars+0] = vx; \
      L[1*nvars+1] = -1.0; \
      L[1*nvars+2] = 0.0; \
      L[1*nvars+3] = 0.0; \
      L[1*nvars+4] = 0; \
      L[2*nvars+0] = -vy; \
      L[2*nvars+1] = 0.0; \
      L[2*nvars+2] = 1.0; \
      L[2*nvars+3] = 0.0; \
      L[2*nvars+4] = 0; \
    } \
    int m_i; \
    for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
      L[m_i*nvars+m_i] = 1.0; \
    } \
  }

/*! \def _NavierStokes3DRightEigenvectors_
  Compute the right eigenvectors, given a solution vector in terms of the conserved variables. The eigenvectors are
  returned as a matrix R whose columns correspond to each eigenvector. The matrix R is stored in the row-major format.
  \n\n
  Reference:
  + Rohde, "Eigenvalues and eigenvectors of the Euler equations in general geometries", AIAA Paper 2001-2609,
    http://dx.doi.org/10.2514/6.2001-2609
*/
#define _NavierStokes3DRightEigenvectors_(u,R,ga,nvars,dir) \
  { \
    _ArraySetValue_(R,nvars*nvars,0.0); \
    double  ga_minus_one = ga-1.0; \
    double  rho,vx,vy,vz,e,P,ek,a,h0; \
    _NavierStokes3DGetFlowVar_(u,rho,vx,vy,vz,e,P,ga); \
    ek   = 0.5 * (vx*vx + vy*vy + vz*vz); \
    a    = sqrt(ga * P / rho); \
    h0   = a*a / ga_minus_one + ek; \
    if (dir == _XDIR_) { \
      R[0*nvars+1] = 1.0; \
      R[1*nvars+1] = vx-a; \
      R[2*nvars+1] = vy; \
      R[3*nvars+1] = vz; \
      R[4*nvars+1] = h0 - a*vx; \
      R[0*nvars+0] = 1.0; \
      R[1*nvars+0] = vx; \
      R[2*nvars+0] = vy; \
      R[3*nvars+0] = vz; \
      R[4*nvars+0] = ek; \
      R[0*nvars+4] = 1.0; \
      R[1*nvars+4] = vx+a; \
      R[2*nvars+4] = vy; \
      R[3*nvars+4] = vz; \
      R[4*nvars+4] = h0 + a*vx; \
      R[0*nvars+2] = 0.0; \
      R[1*nvars+2] = 0.0; \
      R[2*nvars+2] = -1.0; \
      R[3*nvars+2] = 0.0; \
      R[4*nvars+2] = -vy; \
      R[0*nvars+3] = 0.0; \
      R[1*nvars+3] = 0.0; \
      R[2*nvars+3] = 0.0; \
      R[3*nvars+3] = 1.0; \
      R[4*nvars+3] = vz; \
    } else if (dir == _YDIR_) { \
      R[0*nvars+2] = 1.0; \
      R[1*nvars+2] = vx; \
      R[2*nvars+2] = vy-a; \
      R[3*nvars+2] = vz; \
      R[4*nvars+2] = h0 - a*vy; \
      R[0*nvars+0] = 1.0; \
      R[1*nvars+0] = vx; \
      R[2*nvars+0] = vy; \
      R[3*nvars+0] = vz; \
      R[4*nvars+0] = ek; \
      R[0*nvars+4] = 1.0; \
      R[1*nvars+4] = vx; \
      R[2*nvars+4] = vy+a; \
      R[3*nvars+4] = vz; \
      R[4*nvars+4] = h0 + a*vy; \
      R[0*nvars+1] = 0.0; \
      R[1*nvars+1] = 1.0; \
      R[2*nvars+1] = 0.0; \
      R[3*nvars+1] = 0.0; \
      R[4*nvars+1] = vx; \
      R[0*nvars+3] = 0.0; \
      R[1*nvars+3] = 0.0; \
      R[2*nvars+3] = 0.0; \
      R[3*nvars+3] = -1.0; \
      R[4*nvars+3] = -vz; \
    } else if (dir == _ZDIR_) {  \
      R[0*nvars+3] = 1.0; \
      R[1*nvars+3] = vx; \
      R[2*nvars+3] = vy; \
      R[3*nvars+3] = vz-a; \
      R[4*nvars+3] = h0-a*vz; \
      R[0*nvars+0] = 1.0; \
      R[1*nvars+0] = vx; \
      R[2*nvars+0] = vy; \
      R[3*nvars+0] = vz; \
      R[4*nvars+0] = ek; \
      R[0*nvars+4] = 1.0; \
      R[1*nvars+4] = vx; \
      R[2*nvars+4] = vy; \
      R[3*nvars+4] = vz+a; \
      R[4*nvars+4] = h0+a*vz; \
      R[0*nvars+1] = 0.0; \
      R[1*nvars+1] = -1.0; \
      R[2*nvars+1] = 0.0; \
      R[3*nvars+1] = 0.0; \
      R[4*nvars+1] = -vx; \
      R[0*nvars+2] = 0.0; \
      R[1*nvars+2] = 0.0; \
      R[2*nvars+2] = 1.0; \
      R[3*nvars+2] = 0.0; \
      R[4*nvars+2] = vy; \
    } \
    int m_i; \
    for (m_i = _NS3D_NVARS_; m_i < nvars; m_i++) { \
      R[m_i*nvars+m_i] = 1.0; \
    } \
  }

/*! \def _NavierStokes3DCoeffViscosity_
    Compute the viscosity coefficient given the temperature */
#define _NavierStokes3DCoeffViscosity_(mu, T_norm, param) \
  { \
    double T_d = T_norm*param->Tref;  \
    mu = raiseto(T_d/param->T0, 1.5)  \
         * (param->T0 + param->TS)  \
         / (T_d       + param->TS); \
  }

/*! \def _NavierStokes3DCoeffConductivity_
    Compute the conductivity coefficient given the temperature */
#define _NavierStokes3DCoeffConductivity_(kappa, T_norm, param) \
  { \
    double T_d = T_norm*param->Tref; \
    kappa = raiseto(T_d/param->T0, 1.5)                                   \
            * (param->T0 + param->TA * exp(-param->TB/param->T0))       \
            / (T_d       + param->TA * exp(-param->TB/T_d      )); \
  }

/*! \def NavierStokes3D
    \brief Structure containing variables and parameters specific to the 3D Navier Stokes equations.
 *  This structure contains the physical parameters, variables, and function pointers specific to
 *  the 3D Navier-Stokes equations.
*/
/*! \brief Structure containing variables and parameters specific to the 3D Navier Stokes equations.
 *  This structure contains the physical parameters, variables, and function pointers specific to
 *  the 3D Navier-Stokes equations.
*/
typedef struct navierstokes3d_parameters {
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

} NavierStokes3D;

int    NavierStokes3DInitialize (void*,void*);
int    NavierStokes3DCleanup    (void*);

