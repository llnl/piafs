/*! @file Euler1DInitialize.c
    @author Debojyoti Ghosh
    @brief Initialize the 1D Euler equations module.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <physicalmodels/euler1d.h>
#include <mpivars.h>
#include <hypar.h>

double Euler1DComputeCFL (void*,void*,double,double);
int    Euler1DFlux       (double*,double*,int,void*,double);
int    Euler1DSource     (double*,double*,void*,void*,double);

int    Euler1DUpwindRoe     (double*,double*,double*,double*,double*,double*,int,void*,double);
int    Euler1DUpwindRF      (double*,double*,double*,double*,double*,double*,int,void*,double);
int    Euler1DUpwindLLF     (double*,double*,double*,double*,double*,double*,int,void*,double);
int    Euler1DUpwindSWFS    (double*,double*,double*,double*,double*,double*,int,void*,double);
int    Euler1DUpwindRusanov (double*,double*,double*,double*,double*,double*,int,void*,double);

int    Euler1DRoeAverage        (double*,double*,double*,void*);
int    Euler1DLeftEigenvectors  (double*,double*,void*,int);
int    Euler1DRightEigenvectors (double*,double*,void*,int);

int    Euler1DWriteChem (void*,void*,double);
int    Euler1DPreStep (double*,void*,void*,double,double);

/*! Function to initialize the 1D inviscid Euler equations (#Euler1D) module:
    Sets the default parameters, read in and set physics-related parameters,
    and set the physics-related function pointers in #HyPar.

    This file reads the file "physics.inp" that must have the following format:

        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

    where the list of keywords are:

    Keyword name       | Type         | Variable                      | Default value
    ------------------ | ------------ | ----------------------------- | ------------------------
    gamma              | double       | #Euler1D::gamma               | 1.4
    upwinding          | char[]       | #Euler1D::upw_choice          | "roe" (#_ROE_)
    lambda_UB          | double       | #Euler1D::lambda_UB           | 2.48e-7 (248 nm)
    theta              | double       | #Euler1D::theta               | 0.17*pi/180 radians
    f_CO2              | double       | #Euler1D::f_CO2               | 0
    f_O3               | double       | #Euler1D::f_O3                | 0.005
    Ptot               | double       | #Euler1D::Ptot                | 101325 Pa
    Ti                 | double       | #Euler1D::Ti                  | 288 K
    Lz                 | double       | #Euler1D::Lz                  | 0.03 (30 mm)
    z_mm               | double       | #Euler1D::z_mm                | 0
    nz                 | int          | #Euler1D::nz                  | 20
    t_pulse            | int          | #Euler1D::t_pulse             | 1e-8 s (10 nanoseconds)

    \b Note: "physics.inp" is \b optional; if absent, default values will be used.
*/
int Euler1DInitialize(
                      void *s, /*!< Solver object of type #HyPar */
                      void *m  /*!< Object of type #MPIVariables containing MPI-related info */
                     )
{
  HyPar         *solver  = (HyPar*)         s;
  MPIVariables  *mpi     = (MPIVariables*)  m;
  Euler1D       *physics = (Euler1D*)       solver->physics;
  int           ferr;

  static int count = 0;

  if (solver->nvars != _MODEL_NVARS_) {
    fprintf(stderr,"Error in Euler1DInitialize(): nvars has to be %d.\n",_MODEL_NVARS_);
    return(1);
  }
  if (solver->ndims != _MODEL_NDIMS_) {
    fprintf(stderr,"Error in Euler1DInitialize(): ndims has to be %d.\n",_MODEL_NDIMS_);
    return(1);
  }

  /* default values */
  physics->gamma = 1.4;
  strcpy(physics->upw_choice,"roe");
  physics->include_chem = 0;
  physics->chem = NULL;

  /* reading physical model specific inputs */
  char include_chem[_MAX_STRING_SIZE_] = "no";
  if (!mpi->rank) {
    FILE *in;
    if (!count) printf("Reading physical model inputs from file \"physics.inp\".\n");
    in = fopen("physics.inp","r");
    if (!in) printf("Warning: File \"physics.inp\" not found. Using default values.\n");
    else {
      char word[_MAX_STRING_SIZE_];
      ferr = fscanf(in,"%s",word); if (ferr != 1) return(1);
      if (!strcmp(word, "begin")){
        while (strcmp(word, "end")){
          ferr = fscanf(in,"%s",word); if (ferr != 1) return(1);
          if (!strcmp(word, "gamma")) {
            ferr = fscanf(in,"%lf",&physics->gamma);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"upwinding")) {
            ferr = fscanf(in,"%s",physics->upw_choice);
            if (ferr != 1) return(1);
          } else if (!strcmp(word,"include_chemistry")) {
            ferr = fscanf(in,"%s",include_chem);
            if (ferr != 1) return(1);
          } else if (strcmp(word,"end")) {
            char useless[_MAX_STRING_SIZE_];
            ferr = fscanf(in,"%s",useless); if (ferr != 1) return(ferr);
            printf("Warning: keyword %s in file \"physics.inp\" with value %s not ",word,useless);
            printf("recognized or extraneous. Ignoring.\n");
          }
        }
      } else {
        fprintf(stderr,"Error: Illegal format in file \"physics.inp\".\n");
        return(1);
      }
      fclose(in);
    }
  }

  physics->include_chem = (!strcmp(include_chem,"yes"));

#ifndef serial
  MPIBroadcast_character (physics->upw_choice   ,_MAX_STRING_SIZE_,0,&mpi->world);
  MPIBroadcast_double    (&physics->gamma       ,1,0,&mpi->world);
  MPIBroadcast_integer   (&physics->include_chem,1,0,&mpi->world);
#endif

  /* initializing physical model-specific functions */
  solver->PreStep            = Euler1DPreStep;
  solver->ComputeCFL         = Euler1DComputeCFL;
  solver->FFunction          = Euler1DFlux;
  solver->SFunction          = Euler1DSource;
  if      (!strcmp(physics->upw_choice,_ROE_    )) solver->Upwind = Euler1DUpwindRoe;
  else if (!strcmp(physics->upw_choice,_RF_     )) solver->Upwind = Euler1DUpwindRF;
  else if (!strcmp(physics->upw_choice,_LLF_    )) solver->Upwind = Euler1DUpwindLLF;
  else if (!strcmp(physics->upw_choice,_SWFS_   )) solver->Upwind = Euler1DUpwindSWFS;
  else if (!strcmp(physics->upw_choice,_RUSANOV_)) solver->Upwind = Euler1DUpwindRusanov;
  else {
    if (!mpi->rank) fprintf(stderr,"Error in Euler1DInitialize(): %s is not a valid upwinding scheme.\n",
                            physics->upw_choice);
    return(1);
  }
  solver->AveragingFunction     = Euler1DRoeAverage;
  solver->GetLeftEigenvectors   = Euler1DLeftEigenvectors;
  solver->GetRightEigenvectors  = Euler1DRightEigenvectors;
  solver->PhysicsOutput         = Euler1DWriteChem;

  if (physics->include_chem) {
    physics->chem = (Chemistry*) calloc (1, sizeof(Chemistry));
    ChemistryInitialize( solver,
                         physics->chem,
                         mpi,
                         physics->gamma,
                         &physics->L_ref,
                         &physics->v_ref,
                         &physics->t_ref,
                         &physics->P_ref,
                         &physics->rho_ref );
    if (!mpi->rank) {
      printf("Reference quantities:\n");
      printf("    Length: %1.4e (m)\n", physics->L_ref);
      printf("    Time: %1.4e (s)\n", physics->t_ref);
      printf("    Speed: %1.4e (m s^{-1})\n", physics->v_ref);
      printf("    Density: %1.4e (kg m^{-3})\n", physics->rho_ref);
      printf("    Pressure: %1.4e (Pa)\n", physics->P_ref);
    }
  }

  count++;
  return(0);
}
