/*! @file euler1d.h
    @brief 1D Euler Equations (inviscid, compressible flows)
    @author Debojyoti Ghosh

  1D Euler Equations for Inviscid, Compressible Flows\n

  \f{equation}{
    \frac {\partial} {\partial t} \left[\begin{array}{c} \rho \\ \rho u \\ e \end{array}\right]
  + \frac {\partial} {\partial x} \left[\begin{array}{c} \rho u \\ \rho u^2 + p \\ (e+p) u\end{array}\right]
  = \left[\begin{array}{c} 0 \\ 0 \\ \frac{Q}{\gamma-1} \end{array}\right]
  \f}
  where
  \f{equation}{
    e = \frac {p} {\gamma-1} + \frac{1}{2} \rho u^2
  \f}

  Reference for the governing equations:
  + Laney, C. B., Computational Gasdynamics, Cambridge University Press, 1998
*/

#include <basic.h>
#include <math.h>
#include <matops.h>
#include <physicalmodels/chemistry.h>

/*! \def _EULER_1D_
    1D Euler equations
*/
#define _EULER_1D_  "euler1d"

/* define ndims and nvars for this model */
#undef _MODEL_NDIMS_
#undef _EU1D_NVARS_
/*! Number of spatial dimensions */
#define _MODEL_NDIMS_ 1
/*! Number of variables per grid point */
#define _EU1D_NVARS_ 3

/* choices for upwinding schemes */
/*! Roe upwinding scheme */
#define _ROE_     "roe"
/*! Roe-Fixed upwinding scheme (characteristic-based Roe scheme with local Lax-Friedrich entropy fix) */
#define _RF_      "rf-char"
/*! Local Lax-Friedrich upwinding scheme */
#define _LLF_     "llf-char"
/*! Rusanov flux splitting */
#define _RUSANOV_    "rusanov"

/* grid direction */
#undef _XDIR_
/*! dimension corresponding to the \a x spatial dimension */
#define _XDIR_ 0

/*! \def _Euler1DGetFlowVar_
  Compute the flow variables (density, pressure, velocity)
  from the conserved solution vector.
*/
#define _Euler1DGetFlowVar_(u,rho,v,e,P,p) \
  { \
    double gamma = p->gamma; \
    rho = u[0]; \
    v   = u[1] / rho; \
    e   = u[2]; \
    P   = (e - 0.5*rho*v*v) * (gamma-1.0); \
  }

/*! \def _Euler1DSetFlux_
    Set the flux vector given the flow variables (density,
    velocity, pressure).
*/
#define _Euler1DSetFlux_(f,u,p) \
  { \
    double rho,v,e,P; \
    _Euler1DGetFlowVar_(u,rho,v,e,P,p); \
    f[0] = (rho) * (v); \
    f[1] = (rho) * (v) * (v) + (P); \
    f[2] = ((e) + (P)) * (v); \
    int m_i; \
    for (m_i = _EU1D_NVARS_; m_i < p->nvars; m_i++) { \
        f[m_i] = v * u[m_i]; \
    } \
  }

/*! \def _Euler1DRoeAverage_
    Compute the Roe average of two conserved solution
    vectors.
*/
#define _Euler1DRoeAverage_(uavg,uL,uR,p) \
  { \
    double rho ,v ,e ,P ,H ,csq; \
    double rhoL,vL,eL,PL,HL,cLsq; \
    double rhoR,vR,eR,PR,HR,cRsq; \
    double gamma = p->gamma; \
    rhoL = uL[0]; \
    vL   = uL[1] / rhoL; \
    eL   = uL[2]; \
    PL   = (eL - 0.5*rhoL*vL*vL) * (gamma-1.0); \
    cLsq = gamma * PL/rhoL; \
    HL = 0.5*vL*vL + cLsq / (gamma-1.0); \
    rhoR = uR[0]; \
    vR   = uR[1] / rhoR; \
    eR   = uR[2]; \
    PR   = (eR - 0.5*rhoR*vR*vR) * (gamma-1.0); \
    cRsq = gamma * PR/rhoR; \
    HR = 0.5*vR*vR + cRsq / (gamma-1.0); \
    double tL = sqrt(rhoL); \
    double tR = sqrt(rhoR); \
    rho = tL * tR; \
    v   = (tL*vL + tR*vR) / (tL + tR); \
    H   = (tL*HL + tR*HR) / (tL + tR); \
    csq = (gamma-1.0) * (H-0.5*v*v); \
    P   = csq * rho / gamma; \
    e   = P/(gamma-1.0) + 0.5*rho*v*v; \
    \
    uavg[0] = rho; \
    uavg[1] = rho*v; \
    uavg[2] = e; \
    int m_i; \
    for (m_i = _EU1D_NVARS_; m_i < p->nvars; m_i++) { \
        uavg[m_i] = sqrt(uL[m_i]) * sqrt(uR[m_i]); \
    } \
  }

/*! \def _Euler1DEigenvalues_
    Compute the eigenvalues, given the conserved solution vector. The
    eigenvalues are returned as a 3x3 matrix stored in row-major format.
    It is a diagonal matrix with the eigenvalues as diagonal elements.
    Admittedly, it is a wasteful way of storing the eigenvalues.
*/
#define _Euler1DEigenvalues_(u,D,p,dir) \
  { \
    double gamma = p->gamma; \
    double rho,v,e,P,c; \
    _Euler1DGetFlowVar_(u,rho,v,e,P,p); \
    c = sqrt(gamma*P/rho); \
    _ArraySetValue_(D, p->nvars*p->nvars, 0.0); \
    D[0*p->nvars+0] = v; \
    D[1*p->nvars+1] = (v-c); \
    D[2*p->nvars+2] = (v+c); \
    int m_i; \
    for (m_i = _EU1D_NVARS_; m_i < p->nvars; m_i++) { \
      D[m_i*p->nvars+m_i] = v; \
    } \
  }

/*! \def _Euler1DLeftEigenvectors_
    Compute the matrix that has the left-eigenvectors as
    its rows. Stored in row-major format.
*/
#define _Euler1DLeftEigenvectors_(u,L,p,dir) \
  { \
    double gamma = p->gamma; \
    double rho,v,e,P,c; \
    _Euler1DGetFlowVar_(u,rho,v,e,P,p); \
    c    = sqrt(gamma*P/rho); \
    _ArraySetValue_(L, p->nvars*p->nvars, 0.0); \
    L[1*p->nvars+0] = ((gamma - 1)/(rho*c)) * (-(v*v)/2 - c*v/(gamma-1)); \
    L[1*p->nvars+1] = ((gamma - 1)/(rho*c)) * (v + c/(gamma-1)); \
    L[1*p->nvars+2] = ((gamma - 1)/(rho*c)) * (-1); \
    L[0*p->nvars+0] = ((gamma - 1)/(rho*c)) * (rho*(-(v*v)/2+c*c/(gamma-1))/c); \
    L[0*p->nvars+1] = ((gamma - 1)/(rho*c)) * (rho*v/c); \
    L[0*p->nvars+2] = ((gamma - 1)/(rho*c)) * (-rho/c); \
    L[2*p->nvars+0] = ((gamma - 1)/(rho*c)) * ((v*v)/2 - c*v/(gamma-1)); \
    L[2*p->nvars+1] = ((gamma - 1)/(rho*c)) * (-v + c/(gamma-1)); \
    L[2*p->nvars+2] = ((gamma - 1)/(rho*c)) * (1); \
    int m_i; \
    for (m_i = _EU1D_NVARS_; m_i < p->nvars; m_i++) { \
      L[m_i*p->nvars+m_i] = 1.0; \
    } \
  }

/*! \def _Euler1DRightEigenvectors_
    Compute the matrix that has the right-eigenvectors as
    its columns. Stored in row-major format.
*/
#define _Euler1DRightEigenvectors_(u,R,p,dir) \
  { \
    double gamma   = p->gamma; \
    double rho,v,e,P,c; \
    _Euler1DGetFlowVar_(u,rho,v,e,P,p); \
    c    = sqrt(gamma*P/rho); \
    _ArraySetValue_(R, p->nvars*p->nvars, 0.0); \
    R[0*p->nvars+1] = - rho/(2*c);  R[1*p->nvars+1] = -rho*(v-c)/(2*c); R[2*p->nvars+1] = -rho*((v*v)/2+(c*c)/(gamma-1)-c*v)/(2*c);  \
    R[0*p->nvars+0] = 1;            R[1*p->nvars+0] = v;                R[2*p->nvars+0] = v*v / 2;                                   \
    R[0*p->nvars+2] = rho/(2*c);    R[1*p->nvars+2] = rho*(v+c)/(2*c);  R[2*p->nvars+2] = rho*((v*v)/2+(c*c)/(gamma-1)+c*v)/(2*c);   \
    int m_i; \
    for (m_i = _EU1D_NVARS_; m_i < p->nvars; m_i++) { \
      R[m_i*p->nvars+m_i] = 1.0; \
    } \
  }

/*! \def Euler1D
    \brief Structure containing variables and parameters specific to the 1D Euler equations.
 *  This structure contains the physical parameters, variables, and function pointers
 *  specific to the 1D Euler equations.
*/

/*! \brief Structure containing variables and parameters specific to the 1D Euler equations.
 *  This structure contains the physical parameters, variables, and function pointers
 *  specific to the 1D Euler equations.
*/
typedef struct euler1d_parameters {

  double gamma; /*!< Ratio of heat capacities (\f$\gamma\f$) */
  /*! Choice of upwinding scheme.\sa #_ROE_,#_LLF_,#_RF_,#_SWFS_ */
  char upw_choice[_MAX_STRING_SIZE_];

  int include_chem; /*!< Flag to include chemistry */
  void* chem; /*!< Photochemical reactions object */

  int nvars; /*!< Number of variables per grid point */

} Euler1D;

/*! Function to initialize the 1D Euler module */
int    Euler1DInitialize (void*,void*);
/*! Function to clean up the 1D Euler module */
int    Euler1DCleanup    (void*);

