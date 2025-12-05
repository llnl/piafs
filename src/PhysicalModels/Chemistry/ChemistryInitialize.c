/*! @file ChemistryInitialize.c
    @author Debojyoti Ghosh, Albertine Oudin
    @brief Initialize the chemistry module.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <io.h>
#include <physicalmodels/chemistry.h>
#include <mpivars.h>
#include <hypar.h>

void ChemistrySetPhotonDensity(void*,void*,void*,double);

/*! Function to initialize the photochemistry (#Chemistry) module:
    Sets the default parameters, read in and set chemistry-related parameters.

    This file reads the file "chemistry.inp" that must have the following format:

        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

    where the list of keywords are:

    Keyword name       | Type         | Variable                        | Default value
    ------------------ | ------------ | ------------------------------- | ------------------------
    lambda_UV          | double       | #Chemistry::lambda_UV           | 2.48e-7 (248 nm)
    theta              | double       | #Chemistry::theta               | 0.17*pi/180 radians
    f_CO2              | double       | #Chemistry::f_CO2               | 0
    f_O3               | double       | #Chemistry::f_O3                | 0.005
    Ptot               | double       | #Chemistry::Ptot                | 101325 Pa
    Ti                 | double       | #Chemistry::Ti                  | 288 K
    Lz                 | double       | #Chemistry::Lz                  | 0.03 (30 mm)
    z_mm               | double       | #Chemistry::z_mm                | 0
    nz                 | int          | #Chemistry::nz                  | 20
    t_start            | double       | #Chemistry::t_start             | 0.0 s
    t_pulse            | double       | #Chemistry::t_pulse             | 1e-8 s (10 nanoseconds)
    k0a                | double       | #Chemistry::k0a                 | 0.9*3.3e-13 s^{-1}
    k0b                | double       | #Chemistry::k0a                 | 0.1*3.3e-13 s^{-1}
    k1a                | double       | #Chemistry::k1a                 | 0.8*3.95e-13 s^{-1}
    k1b                | double       | #Chemistry::k1a                 | 0.2*3.95e-13 s^{-1}
    k2a                | double       | #Chemistry::k2a                 | 1.2e-16 s^{-1}
    k2b                | double       | #Chemistry::k2a                 | 1.2e-16 s^{-1}
    k3a                | double       | #Chemistry::k3a                 | 1.2e-17 s^{-1}
    k3b                | double       | #Chemistry::k3a                 | 1.0e-17 s^{-1}
    k4                 | double       | #Chemistry::k4                  | 1.1e-16 s^{-1}
    k5                 | double       | #Chemistry::k5                  | 2.0e-16 s^{-1}
    k6                 | double       | #Chemistry::k6                  | 0.2*3.0e-17 s^{-1}
    F0                 | double       | #Chemistry::F0                  | 2000 J/m^2
    sO3                | double       | #Chemistry::sO3                 | 1.1e-21 m^2
    IA                 | double       | #Chemistry::IA                  | 1.0
    IB                 | double       | #Chemistry::IB                  | 1.0
    IC                 | double       | #Chemistry::IC                  | 0.0
    write_all_zlocs    | double       | #Chemistry::write_all_zlocs     | "yes"

    \b Note: "chemistry.inp" is \b optional; if absent, default values will be used.
*/
int ChemistryInitialize( void*  s, /*!< Solver object of type #HyPar */
                         void*  p, /*!< Chemistry object of type #Chemistry */
                         void*  m, /*!< Object of type #MPIVariables containing MPI-related info */
                         const int n_flowvars /*!< Number of flow variables */ )
{
  HyPar         *solver  = (HyPar*)         s;
  MPIVariables  *mpi     = (MPIVariables*)  m;
  Chemistry     *chem    = (Chemistry*)     p;
  int           ferr;

  static int count = 0;

  /* Number of flow variables (Euler/Navier-Stokes) */
  chem->n_flow_vars = n_flowvars;

  /* default values */
  chem->pi = 4.0*atan(1.0);
  chem->NA = 6.02214076e23;
  chem->kB = 1.380649e-23;
  chem->c = 3.0e8; // m s^{-1}
  chem->h = 6.62607015e-34; // J s
  chem->e = 1.60217663e-19; // Coulombs
  chem->M_O2 = 0.032; // [kg]; O2 molar mass
  chem->M_CO2 = 0.044; // [kg]; O2 molar mass
  chem->Cp_O2 = 918.45; // [J kg^{-1} K^{-1}]; O2 Cp
  chem->Cv_O2 = 650.0; // [J kg^{-1} K^{-1}]; O2 Cv
  chem->Cp_CO2 = 842.86; // [J kg^{-1} K^{-1}]; CO2 Cp
  chem->Cv_CO2 = 654.5; // [J kg^{-1} K^{-1}]; CO2 Cv
  chem->mu0_O2 = 1.99e-5; // @275K
  chem->kappa0_O2 = 2.70e-2; // @275K
  chem->mu0_CO2 = 1.45e-5; // @275K
  chem->kappa0_CO2 = 1.58e-2; // @275K

  chem->lambda_UV = 2.48e-7; // [m] (248 nm) - pump wavelength
  chem->theta = 0.17 * chem->pi / 180; // radians; half angle between probe beams

  chem->f_CO2 = 0.0; // CO2 fraction
  chem->f_O3 = 0.005; // O3 fraction
  chem->Ptot = 101325.0; // [Pa]; total gas pressure
  chem->Ti = 288; // [K]; initial temperature

  chem->t_start = 0.0;
  chem->t_pulse = 10*1e-9; // 10 nanoseconds

  double zmax = 0.0;
  if (solver->ndims == 3) {
    // get zmax of the domain
    _GetCoordinate_(_ZDIR_,solver->dim_local[_ZDIR_]-1,solver->dim_local,solver->ghosts,solver->x,zmax);
    MPIMax_double(&zmax, &zmax, 1, &mpi->world);
    chem->Lz = zmax;
    chem->nz = solver->dim_global[_ZDIR_];
  } else {
    chem->Lz = 0.03; // 30 milimeters
    chem->nz = 22;
  }
  chem->z_mm = 0.0;

  chem->k0a = 0.9 * 3.3e-13; // s^{-1}
  chem->k0b = 0.1 * 3.3e-13; // s^{-1}
  chem->k1a = 0.8 * 3.95e-17; // s^{-1}
  chem->k1b = 0.2 * 3.95e-17; // s^{-1}
  chem->k2a = 1.2e-16; // s^{-1}
  chem->k2b = 1.2e-16; // s^{-1}
  chem->k3a = 1.2e-17; // s^{-1}
  chem->k3b = 1.0e-17; // s^{-1}
  chem->k4  = 1.1e-16; // s^{-1}
  chem->k5  = 2.0e-16; // s^{-1}
  chem->k6  = 0.2*3.0e-17; // s^{-1}

  chem->F0 = 2000; // J m^{-2}
  chem->sO3 = 1.1e-21; // m^2

  chem->IA = 1.0;
  chem->IB = 1.0;
  chem->IC = 0.0;

  strcpy(chem->write_all_zlocs, "yes");

  /* reading physical model specific inputs */
  if (!mpi->rank) {
    FILE *in;
    if (!count) printf("Reading chemical reaction inputs from file \"chemistry.inp\".\n");
    in = fopen("chemistry.inp","r");
    if (!in) printf("Warning: File \"chemistry.inp\" not found. Using default values.\n");
    else {
      char word[_MAX_STRING_SIZE_];
      ferr = fscanf(in,"%s",word); if (ferr != 1) return(1);
      if (!strcmp(word, "begin")){
        while (strcmp(word, "end")){
          ferr = fscanf(in,"%s",word); if (ferr != 1) return(1);
          if (!strcmp(word,"lambda_UV")) {
            ferr = fscanf(in,"%lf",&chem->lambda_UV);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"t_start")) {
            ferr = fscanf(in,"%lf",&chem->t_start);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"t_pulse")) {
            ferr = fscanf(in,"%lf",&chem->t_pulse);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"theta")) {
            ferr = fscanf(in,"%lf",&chem->theta);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"f_CO2")) {
            ferr = fscanf(in,"%lf",&chem->f_CO2);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"f_O3")) {
            ferr = fscanf(in,"%lf",&chem->f_O3);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"Ptot")) {
            ferr = fscanf(in,"%lf",&chem->Ptot);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"Ti")) {
            ferr = fscanf(in,"%lf",&chem->Ti);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"Lz")) {
            ferr = fscanf(in,"%lf",&chem->Lz);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"z_mm")) {
            ferr = fscanf(in,"%lf",&chem->z_mm);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"nz")) {
            ferr = fscanf(in,"%d",&chem->nz);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k0a")) {
            ferr = fscanf(in,"%lf",&chem->k0a);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k1a")) {
            ferr = fscanf(in,"%lf",&chem->k1a);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k2a")) {
            ferr = fscanf(in,"%lf",&chem->k2a);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k3a")) {
            ferr = fscanf(in,"%lf",&chem->k3a);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k0b")) {
            ferr = fscanf(in,"%lf",&chem->k0b);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k1b")) {
            ferr = fscanf(in,"%lf",&chem->k1b);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k2b")) {
            ferr = fscanf(in,"%lf",&chem->k2b);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k3b")) {
            ferr = fscanf(in,"%lf",&chem->k3b);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k4")) {
            ferr = fscanf(in,"%lf",&chem->k4);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k5")) {
            ferr = fscanf(in,"%lf",&chem->k5);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"k6")) {
            ferr = fscanf(in,"%lf",&chem->k6);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"F0")) {
            ferr = fscanf(in,"%lf",&chem->F0);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"sO3")) {
            ferr = fscanf(in,"%lf",&chem->sO3);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"IA")) {
            ferr = fscanf(in,"%lf",&chem->IA);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"IB")) {
            ferr = fscanf(in,"%lf",&chem->IB);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"IC")) {
            ferr = fscanf(in,"%lf",&chem->IC);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"write_all_zlocs")) {
            ferr = fscanf(in,"%s",chem->write_all_zlocs);
            if (ferr != 1) return(1);
          } else if (strcmp(word,"end")) {
            char useless[_MAX_STRING_SIZE_];
            ferr = fscanf(in,"%s",useless); if (ferr != 1) return(ferr);
            printf("Warning: keyword %s in file \"chemistry.inp\" with value %s not ",word,useless);
            printf("recognized or extraneous. Ignoring.\n");
          }
        }
      } else {
        fprintf(stderr,"Error: Illegal format in file \"chemistry.inp\".\n");
        return(1);
      }
      fclose(in);
    }
  }

#ifndef serial
  IERR MPIBroadcast_double    (&chem->lambda_UV,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->theta    ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->f_CO2    ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->f_O3     ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->Ptot     ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->Ti       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->Lz       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->z_mm     ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->t_start  ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->t_pulse  ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_integer   (&chem->nz       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k0a      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k1a      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k2a      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k3a      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k0b      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k1b      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k2b      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k3b      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k4       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k5       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->k6       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->F0       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->sO3      ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->IA       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->IB       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_double    (&chem->IC       ,1,0,&mpi->world);                  CHECKERR(ierr);
  IERR MPIBroadcast_character (chem->write_all_zlocs,_MAX_STRING_SIZE_,0,&mpi->world);  CHECKERR(ierr);
#endif

  /* sanity checks */
  if ((chem->f_CO2 > 1.0) || (chem->f_CO2 < 0.0)) {
    fprintf(stderr,"ERROR in ChemistryInitialize(): f_CO2 is not between 0.0 and 1.0 !!!\n");
    return 1;
  }
  if ((chem->f_O3 > 1.0) || (chem->f_O3 < 0.0)) {
    fprintf(stderr,"ERROR in ChemistryInitialize(): f_O3 is not between 0.0 and 1.0 !!!\n");
    return 1;
  }
  if (solver->ndims == 3) {
    if (chem->Lz != zmax) {
      fprintf(stderr,"ERROR in ChemistryInitialize(): grid zmax (%1.2e) is not equal to Lz specified in chemistry.inp (%1.2e).\n", zmax, chem->Lz);
      return 1;
    }
    if (chem->nz != solver->dim_global[_ZDIR_]) {
      fprintf(stderr,"ERROR in ChemistryInitialize(): grid kmax (%d) is not equal to nz specified in chemistry.inp (%d).\n", solver->dim_global[_ZDIR_], chem->nz);
      return 1;
    }
  }

  /* compute O2 fraction */
  chem->f_O2 = 1.0 - chem->f_CO2;

  /* compute gas properties */
  chem->R = chem->NA*chem->kB / (chem->f_O2*chem->M_O2 + chem->f_CO2*chem->M_CO2);
  chem->Cp = chem->f_O2*chem->Cp_O2 + chem->f_CO2*chem->Cp_CO2;
  chem->Cv = chem->f_O2*chem->Cv_O2 + chem->f_CO2*chem->Cv_CO2;
  chem->gamma = chem->Cp / chem->Cv;

  chem->mu0    = chem->f_O2*chem->mu0_O2    + chem->f_CO2*chem->mu0_CO2;
  chem->kappa0 = chem->f_O2*chem->kappa0_O2 + chem->f_CO2*chem->kappa0_CO2;

  /* compute some basic quantities */
  chem->kUV = 2 * chem->pi / chem->lambda_UV;
  chem->kg = 2 * chem->kUV * sin(chem->theta);
  chem->n_O2 = chem->f_O2 * chem->Ptot / (chem->kB * chem->Ti);
  chem->n_O3 = chem->f_O3 * chem->Ptot / (chem->kB * chem->Ti);
  chem->n_CO2 = chem->f_CO2 * chem->Ptot / (chem->kB * chem->Ti);
  chem->cs = sqrt(chem->gamma*chem->R*chem->Ti);

  /* compute reference quantities */
  chem->L_ref = 1.0/chem->kg;
  chem->v_ref = chem->cs;
  chem->t_ref = chem->L_ref / chem->v_ref;
  chem->rho_ref = chem->Ptot/(chem->R*chem->Ti);
  chem->P_ref = chem->rho_ref * chem->v_ref * chem->v_ref;

  /* compute some important stuff */
  chem->t_start_norm = chem->t_start / chem->t_ref;
  chem->t_pulse_norm = chem->t_pulse / chem->t_ref;
  double rate_norm_fac = chem->kg*chem->cs / chem->n_O2;
  chem->k0a_norm = chem->k0a / rate_norm_fac;
  chem->k1a_norm = chem->k1a / rate_norm_fac;
  chem->k2a_norm = chem->k2a / rate_norm_fac;
  chem->k3a_norm = chem->k3a / rate_norm_fac;
  chem->k0b_norm = chem->k0b / rate_norm_fac;
  chem->k1b_norm = chem->k1b / rate_norm_fac;
  chem->k2b_norm = chem->k2b / rate_norm_fac;
  chem->k3b_norm = chem->k3b / rate_norm_fac;
  chem->k4_norm  = chem->k4  / rate_norm_fac;
  chem->k5_norm  = chem->k5  / rate_norm_fac;
  chem->k6_norm  = chem->k6  / rate_norm_fac;
  chem->I0 = chem->F0/chem->t_pulse;
  chem->nu = chem->c / chem->lambda_UV;

  double Ei = chem->Ptot/((chem->gamma-1.0)*chem->n_O2);

  chem->q0a_norm = 0.73 * chem->e/Ei;
  chem->q0b_norm = 2.8  * chem->e/Ei;

  chem->q1a_norm = 0.29 * chem->e/Ei;
  chem->q1b_norm = 0.84 * chem->e/Ei;

  chem->q2a_norm = 4.3  * chem->e/Ei;
  chem->q2b_norm = 0.81 * chem->e/Ei;

  chem->q3a_norm = 0.42 * chem->e/Ei;
  chem->q3b_norm = 1.16 * chem->e/Ei;

  chem->q4_norm  = 1.23 * chem->e/Ei;
  chem->q5_norm  = 1.81 * chem->e/Ei;
  chem->q6_norm  = 0.42 * chem->e/Ei;

  chem->nspecies = 8; // O2, O3, 1D, 1Dg, 3Su, 1Sg, CO2, hnu
  chem->z_i = (solver->ndims == 3 ? -INT_MAX : (int)(ceil(chem->z_mm*0.001 * chem->nz / chem->Lz)));
  int nz = (solver->ndims == 3 ? 1 : chem->z_i + 1);
  chem->n_reacting_species = (chem->nspecies - 1) * nz; // not include hnu
  if (solver->ndims == 3) {
    _GetCoordinate_(_ZDIR_,0,solver->dim_local,solver->ghosts,solver->dxinv,chem->dz);
    chem->dz = 1.0/chem->dz * chem->L_ref;
    chem->z_stride = 0;
    chem->Lz *= chem->L_ref;
  } else {
    chem->dz = chem->Lz / chem->nz;
    chem->z_stride = chem->nspecies - 1;
  }
  chem->grid_stride = chem->n_flow_vars+chem->n_reacting_species;

  if (!mpi->rank) {
    printf("Constants:\n");
    printf("    Avogadro number: %1.4e [m]\n", chem->NA);
    printf("    Boltzmann constant: %1.4e [m]\n", chem->kB);
    printf("    Speed of light: %1.4e [m]\n", chem->c);
    printf("    Planck's constant: %1.4e [m]\n", chem->h);
    printf("    Elementary charge: %1.4e [m]\n", chem->e);
    printf("    O2 molar mass: %1.4e [kg]\n", chem->M_O2);
    printf("    CO2 molar mass: %1.4e [kg]\n", chem->M_CO2);
    printf("    Specific gas constant: %1.4e [m]\n", chem->R);
    printf("    Specific heat ratio: %1.4e\n", chem->gamma);
    printf("    Reference viscosity of  O2 @275K: %1.4e [kg m^{-1} s^{-1}]\n", chem->mu0_O2 );
    printf("    Reference viscosity of CO2 @275K: %1.4e [kg m^{-1} s^{-1}]\n", chem->mu0_CO2);
    printf("    Reference conductivity of  O2 @275K: %1.4e [W m^{-1} K^{-1}]\n", chem->kappa0_O2 );
    printf("    Reference conductivity of CO2 @275K: %1.4e [W m^{-1} K^{-1}]\n", chem->kappa0_CO2);
    printf("    Reference viscosity of gas mixture @275K: %1.4e [kg m^{-1} s^{-1}]\n", chem->mu0 );
    printf("    Reference conductivity of gas mixture @275K: %1.4e [kg m^{-1} s^{-1}]\n", chem->kappa0 );
    printf("Photo-Chemistry:\n");
    printf("    Pump wavelength: %1.4e [m]\n", chem->lambda_UV);
    printf("    Beam half angle: %1.4e [radians]\n", chem->theta);
    printf("    Pump beam wavenumber: %1.4e [m^{-1}]\n", chem->kUV);
    printf("    Grating wavenumber: %1.4e [m^{-1}]\n", chem->kg);
    printf("    O2 fraction: %1.4e\n", chem->f_O2);
    printf("    O3 fraction: %1.4e\n", chem->f_O3);
    printf("    CO2 fraction: %1.4e\n", chem->f_CO2);
    printf("    Pressure: %1.4e [Pa]\n", chem->Ptot);
    printf("    Temperature: %1.4e [K]\n", chem->Ti);
    printf("    O2 number density:  %1.4e [m^{-3}]\n", chem->n_O2);
    printf("    O3 number density:  %1.4e [m^{-3}]\n", chem->n_O3);
    printf("    CO2 number density: %1.4e [m^{-3}]\n", chem->n_CO2);
    printf("    Sound speed: %1.4e [m s^{-1}]\n", chem->cs);
    printf("    Pulse start time: %1.4e (s), %1.4e (normalized)\n",
                chem->t_start, chem->t_start_norm );
    printf("    Pulse duration: %1.4e (s), %1.4e (normalized)\n",
                chem->t_pulse, chem->t_pulse_norm );
    printf("    Gas length: %1.4e [m]\n", chem->Lz);
    if (solver->ndims < 3) {
      printf("    Number of z-layers: %d\n", chem->nz);
      printf("    Z-location: %1.4e [m] (z_i = %d)\n", chem->z_mm*1e-3,chem->z_i);
    }
    printf("    lambda_ac (acoustic spatial period): %1.4e (m)\n", (2*chem->pi/chem->kg) );
    printf("      normalized lambda_ac: %1.4e\n", (2*chem->pi/chem->kg)/chem->L_ref );
    printf("    tau_ac (acoustic time period): %1.4e (s)\n", (2*chem->pi/chem->kg)/chem->cs );
    printf("      normalized tau_ac: %1.4e (s)\n", ((2*chem->pi/chem->kg)/chem->cs)/chem->t_ref );
    printf("    Reaction rates:\n");
    printf("        k0a = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k0a, chem->k0a_norm);
    printf("        k0b = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k0b, chem->k0b_norm);
    printf("        k1a = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k1a, chem->k1a_norm);
    printf("        k1b = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k1b, chem->k1b_norm);
    printf("        k2a = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k2a, chem->k2a_norm);
    printf("        k2b = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k2b, chem->k2b_norm);
    printf("        k3a = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k3a, chem->k3a_norm);
    printf("        k3b = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k3b, chem->k3b_norm);
    printf("        k4  = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k4 , chem->k4_norm );
    printf("        k5  = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k5 , chem->k5_norm );
    printf("        k6  = %1.4e (m^3 s^{-1}), %1.4e (normalized)\n", chem->k6 , chem->k6_norm );
    printf("    Heating rates (normalized):\n");
    printf("        q0a = %1.4e\n", chem->q0a_norm);
    printf("        q0b = %1.4e\n", chem->q0b_norm);
    printf("        q1a = %1.4e\n", chem->q1a_norm);
    printf("        q1b = %1.4e\n", chem->q1b_norm);
    printf("        q2a = %1.4e\n", chem->q2a_norm);
    printf("        q2b = %1.4e\n", chem->q2b_norm);
    printf("        q3a = %1.4e\n", chem->q3a_norm);
    printf("        q3b = %1.4e\n", chem->q3b_norm);
    printf("        q4  = %1.4e\n", chem->q4_norm );
    printf("        q5  = %1.4e\n", chem->q5_norm );
    printf("        q6  = %1.4e\n", chem->q6_norm );
    printf("    F0: %1.4e [J m^{-2}]\n", chem->F0);
    printf("    I0: %1.4e [J m^{-2} s^{-1}]\n", chem->I0);
    printf("    Intensity function parameters: %1.4e, %1.4e, %1.4e\n", chem->IA, chem->IB, chem->IC);
    printf("        IO * (IA + IB * cos( kg * x * (1-IC*x) ) )\n");
    printf("    nu: %1.4e [s^{-1}]\n", chem->nu);
    printf("    Ozone absorbtion cross-section: %1.4e [m^2]\n", chem->sO3);
    printf("Reference quantities:\n");
    printf("    Length: %1.4e (m)\n", chem->L_ref);
    printf("    Time: %1.4e (s)\n", chem->t_ref);
    printf("    Speed: %1.4e (m s^{-1})\n", chem->v_ref);
    printf("    Density: %1.4e (kg m^{-3})\n", chem->rho_ref);
    printf("    Pressure: %1.4e (Pa)\n", chem->P_ref);
    printf("    Temperature: %1.4e (Pa)\n", chem->Ti);
  }

  // allocate arrays
  chem->nv_hnu   = (double*) calloc (solver->npoints_local_wghosts*nz, sizeof(double));

  /* allocate array to hold the beam intensity field */
  chem->imap = (double*) calloc (solver->npoints_local_wghosts, sizeof(double));
  /* read beam intensity from provided file, if available */
  int read_flag = 0;
  char fname_root[_MAX_STRING_SIZE_] = "imap";
  if (!mpi->rank) {
    printf("ChemistryInitialize(): reading intensity map from file.\n");
  }
  ReadArray( solver->ndims, 1, solver->dim_global, solver->dim_local, solver->ghosts,
             solver, mpi, NULL, chem->imap, fname_root, &read_flag);
  if (!read_flag) {

    if (!mpi->rank) {
      printf("ChemistryInitialize(): could not read intensity map from file; setting it as:.\n");
      printf("ChemistryInitialize():      IA + IB * cos( kg * x * (1-IC*x) )                \n");
    }

    // get xmin of the domain
    double x0 = 0.0;
    _GetCoordinate_(0,0,solver->dim_local,solver->ghosts,solver->x,x0);
    MPIMin_double(&x0, &x0, 1, &mpi->world);

    int index[solver->ndims];
    int done = 0; _ArraySetValue_(index,solver->ndims,0);
    while (!done) {
      int p; _ArrayIndex1D_(solver->ndims,solver->dim_local,index,solver->ghosts,p);
      double x; _GetCoordinate_(0,index[0],solver->dim_local,solver->ghosts,solver->x,x);
      chem->imap[p] = chem->IA + chem->IB * cos( chem->kg * chem->L_ref*(x-x0)*(1.0-chem->IC*(x-x0)) );
      _ArrayIncrementIndex_(solver->ndims,solver->dim_local,index,done);

    }
  }

  count++;
  return 0;
}
