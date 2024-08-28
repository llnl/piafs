Numerical Method
================

HyPar solves the following partial differential equation (PDE) using a conservative finite-difference
algorithm on a Cartesian grid.
\f{equation}{
  \frac {\partial {\bf u}} {\partial t} = {\bf F}_{\rm hyp}\left({\bf u}\right) + {\bf F}_{\rm par}\left({\bf u}\right) + {\bf F}_{\rm sou}\left({\bf u}\right)
\f}
where \f${\bf F}_{\rm hyp}\f$ is the hyperbolic term, \f${\bf F}_{\rm par}\f$ is the parabolic term, and 
\f${\bf F}_{\rm sou}\f$ is the source term. Each of these is discretized in space as described below (in
the section "Spatial Discretization"), to 
obtain the following semi-discrete ordinary differential equation (ODE) in time:
\f{equation}{
  \frac {d {\bf u}} {d t} = \hat{\bf F}_{\rm hyp}\left({\bf u}\right) + \hat{\bf F}_{\rm par}\left({\bf u}\right) + \hat{\bf F}_{\rm sou}\left({\bf u}\right)
\f}
where \f$\hat{\left(\cdot\right)}\f$ represents the spatially discretized terms. The governing PDE can be
of any space dimension. 

\section spatial_discretization Spatial Discretization

Hyperbolic term
---------------

The hyperbolic term is of the following form:
\f{equation}{
  {\bf F}_{\rm hyp}\left({\bf u}\right) = -\sum_{d=0}^{D-1} \frac{\partial {\bf f}_d\left({\bf u}\right)}{\partial x_d}
\f}
and is discretized as:
\f{equation}{
  {\bf F}_{\rm hyp}\left({\bf u}\right) \approx - \sum_{d=0}^{D-1} \frac{1}{\Delta x_d} \left[ \hat{\bf f}_{d,j+1/2} - \hat{\bf f}_{d,j-1/2} \right]
\f}
where \f$d\f$ is the spatial dimension, \f$D\f$ is the total number of spatial dimensions, \f$j\f$ denotes the grid index along \f$d\f$. This
is implemented in HyperbolicFunction().

The numerical approximation \f$\hat{\bf f}_{d,j+1/2}\f$ of the primitive of the flux \f${\bf f}_d\left({\bf u}\right)\f$ at the grid interfaces \f$j+1/2\f$ is expressed
as
\f{equation}{
  \hat{\bf f}_{d,j+1/2} = \mathcal{U}\left(\hat{\bf f}^L_{d,j+1/2},\hat{\bf f}^R_{d,j+1/2},\hat{\bf u}^L_{d,j+1/2},\hat{\bf u}^R_{d,j+1/2}\right)
\f}
where \f$\mathcal{U}\f$ is an upwinding function. #HyPar::Upwind points to the physical model-specific upwinding function that implements
\f$\mathcal{U}\f$, and is set by the initialization function of a specific physical model (for example Euler1DInitialize()). The physical model
is specified by setting #HyPar::model (read from \a "solver.inp" in ReadInputs()).

\f$\hat{\bf f}^{L,R}_{d,j+1/2}\f$ are the left- and right-biased numerically interpolated values of the primitive of the flux \f${\bf f}_d\left({\bf u}\right)\f$
at the grid interfaces and are computed using #HyPar::InterpolateInterfacesHyp. They are initialized in InitializeSolvers() based on the value of 
#HyPar::spatial_scheme_hyp (read from \a "solver.inp" in ReadInputs()). See interpolation.h for all the spatial discretization schemes implemented.

#HyPar::HyperbolicFunction points to HyperbolicFunction().

Parabolic term
--------------

The parabolic term is of the following form:
\f{equation}{
  {\bf F}_{\rm par}\left({\bf u}\right) = \sum_{d1=0}^{D-1}\sum_{d2=0}^{D-1} \frac {\partial^2 h_{d1,d2}\left({\bf u}\right)} {\partial x_{d1} \partial x_{d2}}
\f}
where \f$d1,d2\f$ are spatial dimensions, \f$D\f$ is the total number of spatial dimensions. 

The parabolic term is discretized as
\f{equation}{
  {\bf F}_{\rm par}\left({\bf u}\right) \approx \sum_{d1=0}^{D-1}\sum_{d2=0}^{D-1} \frac {1}{\Delta x_{d1} \Delta x_{d2}} \left[ \mathcal{D}_{d1}\left(\mathcal{D}_{d2}\left({\bf g}_d\right)\right) \right]
\f}
  where \f$\mathcal{D}_d\f$ denotes the finite-difference first derivative operator along spatial dimension \f$d\f$, and is computed using #HyPar::FirstDerivativePar.

The function pointer #HyPar::FirstDerivativePar are set in InitializeSolvers() based on the value of 
#HyPar::spatial_scheme_par (read from \a "solver.inp" in ReadInputs()). See firstderivative.h for the spatial
disretization methods implemented.

Source term
-----------

#HyPar::SourceFunction points to SourceFunction(). There is no discretization involved in general, and this function
just calls the physical model-specific source function to which #HyPar::SFunction points.


\section time_integration Time Integration

The semi-discrete ODE is integrated in time using explicit multi-stage time integration methods. The ODE can be written as:
\f{equation}{
  \frac {d {\bf u}} {d t} = {\bf F}\left({\bf u}\right)
\f}
where
\f{equation}{
  {\bf F}\left({\bf u}\right) = \hat{\bf F}_{\rm hyp}\left({\bf u}\right) + \hat{\bf F}_{\rm par}\left({\bf u}\right) + \hat{\bf F}_{\rm sou}\left({\bf u}\right)
\f}
The following explicit time integration methods are implemented in HyPar (see timeintegration.h):
+ Forward Euler - TimeForwardEuler(), #_FORWARD_EULER_
+ Explicit Runge-Kutta - TimeRK(), #_RK_

\sa Solve()
