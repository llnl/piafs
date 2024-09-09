/*! @file chemistry.h
    @brief PIAFS Chemistry
    @author Debojyoti Ghosh, Albertine Oudin

    Structures/data for the photochemical reactions
*/

#ifndef _CHEM_H_
#define _CHEM_H_

#include <basic.h>
#include <string.h>

/*! \def _ChemistrySetRHS_
    Set the RHS of the reaction equations given the
    number densities of reacting species
*/
#define _ChemistrySetRHS_(f, u, chem) \
  { \
    double n_O2 = u[0]; \
    double n_O3 = u[1]; \
    double n_1D = u[2]; \
    double n_1Dg = u[3]; \
    double n_1Sg = u[4]; \
    double n_hnu = u[5]; \
    \
    double k0a = chem->k0a_norm; \
    double k0b = chem->k0b_norm; \
    double k1a = chem->k1a_norm; \
    double k1b = chem->k1b_norm; \
    double k2a = chem->k2a_norm; \
    double k2b = chem->k2b_norm; \
    \
    /* O3  */ f[0] = -(k0a+k0b)*n_hnu*n_O3 - (k2a+k2b)*n_O3*n_1D; \
    /* 1D  */ f[1] = k0a*n_hnu*n_O3 - (k1a+k1b)*n_1D*n_O2; \
    /* 1Dg */ f[2] = k0a*n_hnu*n_O3 + k1b*n_1D*n_O2; \
    /* 1Sg */ f[3] = k1a*n_1D*n_O2; \
  }

/*! \def _ChemistrySetQ_
    Set the Q for the Euler/NS equations given the
    number densities of reacting species
*/
#define _ChemistrySetQ_(Q, u, chem) \
  { \
    double n_O2 = u[0]; \
    double n_O3 = u[1]; \
    double n_1D = u[2]; \
    double n_1Dg = u[3]; \
    double n_1Sg = u[4]; \
    double n_hnu = u[5]; \
    \
    double k0a = chem->k0a_norm; \
    double k0b = chem->k0b_norm; \
    double k1a = chem->k1a_norm; \
    double k1b = chem->k1b_norm; \
    double k2a = chem->k2a_norm; \
    double k2b = chem->k2b_norm; \
    \
    double q0a = chem->q0a_norm; \
    double q0b = chem->q0b_norm; \
    double q1a = chem->q1a_norm; \
    double q1b = chem->q1b_norm; \
    double q2a = chem->q2a_norm; \
    double q2b = chem->q2b_norm; \
    \
    Q =   (q0a*k0a +q0b*k0b) * n_hnu * n_O3 \
        + (q1a*k1a +q1b*k1b) * n_O2 * n_1D \
        + (q2a*k2a +q2b*k2b) * n_O3 * n_1D; \
  }

/*! \def Chemistry
    \brief Structure containing variables and parameters specific to the photochemical reactions
 *  This structure contains the physical parameters, variables, and function pointers
 *  specific to the photochemical reactions
*/

/*! \brief Structure containing variables and parameters specific to the photochemical reactions.
 *  This structure contains the physical parameters, variables, and function pointers
 *  specific to the photochemical reactions.
*/
typedef struct chemistry_parameters {

  // Some constants
  double pi; /*!< Pi */
  double NA; /*!< Avogadro's number */
  double kB; /*!< Boltzman's constant */
  double c; /*!< speed of light */
  double h; /*!< Planck's constant */
  double e; /*!< elementary charge */
  double R; /*!< specific gas constant */

  // Physical setup;
  double lambda_UV; /*!< pump wavelength in meters */
  double theta; /*!< half angle between probe beams in radians */
  double kUV; /*!<pump beam wave vector */
  double kg; /*!< grating wave vector */

  double f_CO2; /*!< CO2 fraction */
  double f_O2; /*!< O2 fraction */
  double f_O3; /*!< O3 fraction */
  double Ptot; /*!< total gas pressure in Pascals */
  double Ti; /*!< initial temperature in Kelvin */
  double M_O2; /*!< O2 molar mass in kg */
  double n_O2; /*!< initial O2 concentration in m^{-3} */
  double n_O3; /*!< initial O3 concentration in m^{-3} */
  double rho_O2; /*!< O2 density */
  double cs; /*!< speed of sound */

  // reference quantities for normalization
  double L_ref; /*!< reference length */
  double v_ref; /*!< reference speed */
  double t_ref; /*!< reference time */
  double rho_ref; /*!< reference density */
  double P_ref; /*!< reference pressure */

  /* heating and reaction arrays */
  double t_pulse; /*!< Duration of pulse */
  double t_pulse_norm; /*!< Normalized duration of pulse */
  double Lz; /*!< Gas length */
  int nz; /*!< number of z-layers */
  double dz; /*!< grid spacing along z */
  double z_mm; /*!< z-location to simulate (in milimeters) */
  int z_i; /*!< z-index to simulate at */

  // reaction rates
  double k0a; /*!< reaction rate */
  double k0b; /*!< reaction rate */
  double k1a; /*!< reaction rate */
  double k1b; /*!< reaction rate */
  double k2a; /*!< reaction rate */
  double k2b; /*!< reaction rate */
  double k0a_norm; /*!< normalized reaction rate */
  double k0b_norm; /*!< normalized reaction rate */
  double k1a_norm; /*!< normalized reaction rate */
  double k1b_norm; /*!< normalized reaction rate */
  double k2a_norm; /*!< normalized reaction rate */
  double k2b_norm; /*!< normalized reaction rate */

  // heating rates
  double q0a_norm; /*!< normalized heating rate */
  double q0b_norm; /*!< normalized heating rate */
  double q1a_norm; /*!< normalized heating rate */
  double q1b_norm; /*!< normalized heating rate */
  double q2a_norm; /*!< normalized heating rate */
  double q2b_norm; /*!< normalized heating rate */

  // beam parameters
  double F0; /*!< Fluence [J/m^2] */
  double I0; /*!< intensity */
  double nu; /*!< */
  double sO3; /*!< Ozone absorbtion cross-section [m^2] */

  int nspecies; /*!< number of reacting species */
  double* nv_O2; /*!< number density of O2 */
  double* nv_O3; /*!< number density of O3 */
  double* nv_O3old; /*!< number density of O3 (previous timestep) */
  double* nv_1D; /*!< number density of O(1D) */
  double* nv_1Dg; /*!< number density of O(1-Delta_g) */
  double* nv_1Sg; /*!< number density of O(1-Sigma_g ) */
  double* nv_hnu; /*!< number density of h-nu (photons) */

  double* Qv; /*!< heating source term for Euler/NS equations */

  char ti_scheme[_MAX_STRING_SIZE_]; /*!< time integrator to use for reaction equations */

} Chemistry;

/*! Function to initialize the chemistry object */
int ChemistryInitialize(void*,void*,void*,double);
/*! Function to cleanup the chemistry object */
int ChemistryCleanup(void*);
/*! Function to write reacting species to file */
int ChemistryWriteSpecies(void*,void*,void*,double);
/*! Solve the reaction equations */
int ChemistrySolve(void*,void*,void*,double,double);

#endif
