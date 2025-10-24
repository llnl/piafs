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
    double n_O2  = u[0]; \
    double n_O3  = u[1]; \
    double n_1D  = u[2]; \
    double n_1Dg = u[3]; \
    double n_3Su = u[4]; \
    double n_1Sg = u[5]; \
    double n_CO2 = u[6]; \
    double n_hnu = u[7]; \
    \
    double k0a = chem->k0a_norm; \
    double k0b = chem->k0b_norm; \
    double k1a = chem->k1a_norm; \
    double k1b = chem->k1b_norm; \
    double k2a = chem->k2a_norm; \
    double k2b = chem->k2b_norm; \
    double k3a = chem->k3a_norm; \
    double k3b = chem->k3b_norm; \
    double k4  = chem->k4_norm; \
    double k5  = chem->k5_norm; \
    double k6  = chem->k6_norm; \
    \
    /* O3  */ f[0] = - (k0a+k0b) * n_hnu * n_O3  \
                     - (k2a+k2b) * n_O3  * n_1D  \
                     - (k3a+k3b) * n_O3  * n_1Sg \
                     - k5        * n_O3  * n_3Su \
                     - k6        * n_1Dg * n_O3; \
    \
    /* 1D  */ f[1] =   k0a       * n_hnu * n_O3   \
                     - (k1a+k1b) * n_1D  * n_O2   \
                     - (k2a+k2b) * n_1D  * n_O3   \
                     - k4        * n_1D  * n_CO2; \
    \
    /* 1Dg */ f[2] =    k0a * n_hnu * n_O3  \
                      + k5  * n_O3  * n_3Su \
                      - k6  * n_1Dg * n_O3; \
    \
    /* 3Su */ f[3] =    k2a * n_O3 * n_1D   \
                      - k5  * n_O3 * n_3Su; \
    \
    /* 1Sg */ f[4] =    k1a       * n_1D * n_O2 \
                      - (k3a+k3b) * n_O3 * n_1Sg; \
    \
    /* CO2 */ f[5] = 0.0; \
  }

/*! \def _ChemistrySetQ_
    Set the Q for the Euler/NS equations given the
    number densities of reacting species
*/
#define _ChemistrySetQ_(Q, u, chem) \
  { \
    double n_O2  = u[0]; \
    double n_O3  = u[1]; \
    double n_1D  = u[2]; \
    double n_1Dg = u[3]; \
    double n_3Su = u[4]; \
    double n_1Sg = u[5]; \
    double n_CO2 = u[6]; \
    double n_hnu = u[7]; \
    \
    double k0a = chem->k0a_norm; \
    double k0b = chem->k0b_norm; \
    double k1a = chem->k1a_norm; \
    double k1b = chem->k1b_norm; \
    double k2a = chem->k2a_norm; \
    double k2b = chem->k2b_norm; \
    double k3a = chem->k3a_norm; \
    double k3b = chem->k3b_norm; \
    double k4  = chem->k4_norm; \
    double k5  = chem->k5_norm; \
    double k6  = chem->k6_norm; \
    \
    double q0a = chem->q0a_norm; \
    double q0b = chem->q0b_norm; \
    double q1a = chem->q1a_norm; \
    double q1b = chem->q1b_norm; \
    double q2a = chem->q2a_norm; \
    double q2b = chem->q2b_norm; \
    double q3a = chem->q3a_norm; \
    double q3b = chem->q3b_norm; \
    double q4  = chem->q4_norm; \
    double q5  = chem->q5_norm; \
    double q6  = chem->q6_norm; \
    \
    Q =   (q0a*k0a +q0b*k0b) * n_hnu * n_O3  \
        + (q1a*k1a +q1b*k1b) * n_O2  * n_1D  \
        + (q2a*k2a +q2b*k2b) * n_O3  * n_1D  \
        + (q3a*k3a +q3b*k3b) * n_O3  * n_1Sg \
        + q4*k4              * n_1D  * n_CO2 \
        + q5*k5              * n_O3  * n_3Su \
        + q6*k6              * n_1Dg * n_O3; \
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
  double pi;      /*!< Pi */
  double NA;      /*!< Avogadro's number */
  double kB;      /*!< Boltzman's constant */
  double c;       /*!< speed of light */
  double h;       /*!< Planck's constant */
  double e;       /*!< elementary charge */
  double Cp_O2;   /*!< Heat capacity for O2 */
  double Cv_O2;   /*!< Heat capacity for O2 */
  double Cp_CO2;  /*!< Heat capacity for CO2 */
  double Cv_CO2;  /*!< Heat capacity for CO2 */
  double Cp;      /*!< Heat capacity */
  double Cv;      /*!< Heat capacity */
  double R;       /*!< specific gas constant */
  double gamma;   /*!< specific heat ratio */

  double mu0_O2; // Reference viscosity for O2 @275K
  double kappa0_O2; // Reference conductivity for CO2 @275K
  double mu0_CO2; // Reference viscosity for CO2 @275K
  double kappa0_CO2; // Reference conductivity for CO2 @275K
  double mu0; // reference viscosity @275K
  double kappa0; // reference conductivity @275K

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
  double M_CO2; /*!< CO2 molar mass in kg */
  double n_O2; /*!< initial O2 concentration in m^{-3} */
  double n_O3; /*!< initial O3 concentration in m^{-3} */
  double n_CO2; /*!< initial CO2 concentration in m^{-3} */
  double cs; /*!< speed of sound */

  // reference quantities for normalization
  double L_ref; /*!< reference length */
  double v_ref; /*!< reference speed */
  double t_ref; /*!< reference time */
  double rho_ref; /*!< reference density */
  double P_ref; /*!< reference pressure */

  /* heating and reaction arrays */
  double t_start; /*!< Starting time of pulse */
  double t_pulse; /*!< Duration of pulse */
  double t_start_norm; /*!< Normalized starting time of pulse */
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
  double k3a; /*!< reaction rate */
  double k3b; /*!< reaction rate */
  double k4;  /*!< reaction rate */
  double k5;  /*!< reaction rate */
  double k6;  /*!< reaction rate */
  double k0a_norm; /*!< normalized reaction rate */
  double k0b_norm; /*!< normalized reaction rate */
  double k1a_norm; /*!< normalized reaction rate */
  double k1b_norm; /*!< normalized reaction rate */
  double k2a_norm; /*!< normalized reaction rate */
  double k2b_norm; /*!< normalized reaction rate */
  double k3a_norm; /*!< normalized reaction rate */
  double k3b_norm; /*!< normalized reaction rate */
  double k4_norm;  /*!< normalized reaction rate */
  double k5_norm;  /*!< normalized reaction rate */
  double k6_norm;  /*!< normalized reaction rate */

  // heating rates
  double q0a_norm; /*!< normalized heating rate */
  double q0b_norm; /*!< normalized heating rate */
  double q1a_norm; /*!< normalized heating rate */
  double q1b_norm; /*!< normalized heating rate */
  double q2a_norm; /*!< normalized heating rate */
  double q2b_norm; /*!< normalized heating rate */
  double q3a_norm; /*!< normalized heating rate */
  double q3b_norm; /*!< normalized heating rate */
  double q4_norm;  /*!< normalized heating rate */
  double q5_norm;  /*!< normalized heating rate */
  double q6_norm;  /*!< normalized heating rate */

  // beam parameters
  double F0; /*!< Fluence [J/m^2] */
  double I0; /*!< intensity */
  double IA; /*!< intensity function parameter */
  double IB; /*!< intensity function parameter */
  double IC; /*!< intensity function parameter */
  double nu; /*!< */
  double sO3; /*!< Ozone absorbtion cross-section [m^2] */
  double* imap; /*!< Array of intensity values read in from file */

  int n_flow_vars; /*!< number of flow variables */
  int nspecies; /*!< number of species */
  int n_reacting_species; /*!< number of reacting species that advects with flow */
  double* nv_O3old; /*!< number density of O3 (previous timestep) */
  double* nv_hnu; /*!< number density of h-nu (photons) */

  int grid_stride; /*!< grid stride */
  int z_stride; /*!< z stride */

  char write_all_zlocs[_MAX_STRING_SIZE_]; /*!< write chemical species at all z-locations? */

} Chemistry;

/*! Function to initialize the chemistry object */
int ChemistryInitialize(void*,void*,void*,int);
/*! Function to cleanup the chemistry object */
int ChemistryCleanup(void*);
/*! Function to write reacting species to file */
int ChemistryWriteSpecies(void*,double*,void*,void*,double);
/*! Set the reaction source terms */
int ChemistrySource(void*,double*,double*,void*,void*,double);
/*! Pre-time-step operations */
int ChemistryPreStep(void*,double*,void*);

#endif
