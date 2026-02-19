# Numerical Methods

PIAFS uses a conservative finite-difference approach on Cartesian grids.

## Governing Equations

PIAFS solves the following partial differential equation:

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{F}_{\rm hyp}(\mathbf{u}) + \mathbf{F}_{\rm par}(\mathbf{u}) + \mathbf{F}_{\rm sou}(\mathbf{u})
$$

Where:
* $\mathbf{F}_{\rm hyp}$ - Hyperbolic (inviscid) term
* $\mathbf{F}_{\rm par}$ - Parabolic (viscous) term
* $\mathbf{F}_{\rm sou}$ - Source term

## Spatial Discretization

### Hyperbolic Term

The hyperbolic term is discretized as:

$$
\mathbf{F}_{\rm hyp}(\mathbf{u}) \approx -\sum_{d=0}^{D-1} \frac{1}{\Delta x_d} \left[ \hat{\mathbf{f}}_{d,j+1/2} - \hat{\mathbf{f}}_{d,j-1/2} \right]
$$

Interface fluxes are computed using upwinding with reconstructed left/right states.

**Available Schemes:**
* 1st order upwind
* 2nd order MUSCL with limiters
* 3rd order MUSCL
* 5th order WENO
* 5th order CRWENO (compact)
* 5th order HCWENO (hybrid compact)

### Parabolic Term

The parabolic term uses centered finite differences:

$$
\mathbf{F}_{\rm par}(\mathbf{u}) \approx \sum_{d_1=0}^{D-1}\sum_{d_2=0}^{D-1} \frac{1}{\Delta x_{d_1} \Delta x_{d_2}} \left[ \mathcal{D}_{d_1}(\mathcal{D}_{d_2}(\mathbf{g}_d)) \right]
$$

**Available Schemes:**
* 2nd order central difference
* 4th order central difference

## Time Integration

The semi-discrete ODE is integrated using explicit methods:

$$
\frac{d \mathbf{u}}{d t} = \mathbf{F}(\mathbf{u})
$$

**Available Methods:**
* Forward Euler (1st order)
* Runge-Kutta 2nd order (RK2)
* Runge-Kutta 3rd order (RK3, SSP)
* Runge-Kutta 4th order (RK4)

## Upwinding Methods

For the Euler/Navier-Stokes equations:

* **Roe** - Roe's approximate Riemann solver (default)
* **Rusanov** - Local Lax-Friedrichs
* **LLF** - Local Lax-Friedrichs

## WENO Schemes

WENO (Weighted Essentially Non-Oscillatory) schemes provide high-order accuracy with shock-capturing capability.

Key parameters (in `weno.inp`):
* `epsilon` - Small number for stability (default: 1e-6)
* `p` - Power in smoothness indicator (default: 2.0)
* `mapped` - Use mapped WENO weights
* `borges` - Use improved Borges weights

## MUSCL Schemes

MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) with limiters.

**Limiters:**
* GeneralizedMinMod (GMM)
* MinMod
* Van Leer
* SuperBee

Configure in `muscl.inp`.

## Compact Schemes

Compact finite-difference schemes require solving tridiagonal systems.

Configured in `lusolver.inp` with parameters for iterative solvers.

## Conservation

PIAFS is designed to conserve:
* Mass
* Momentum
* Total energy

Conservation checks can be enabled with `conservation_check yes` in `solver.inp`.

## References

For detailed mathematical formulations and references, see the PIAFS publications and the original Doxygen documentation.
