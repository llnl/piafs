/*! @file NavierStokes2DInitialize.c
    @author Debojyoti Ghosh
    @brief Initialization of the physics-related variables and function pointers for the 2D Navier-Stokes system
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <boundaryconditions.h>
#include <physicalmodels/navierstokes2d.h>
#include <mpivars.h>
#include <hypar.h>

double NavierStokes2DComputeCFL (void*,void*,double,double);

int NavierStokes2DFlux              (double*,double*,int,void*,double);
int NavierStokes2DRoeAverage        (double*,double*,double*,void*);
int NavierStokes2DParabolicFunction (double*,double*,void*,void*,double);
int NavierStokes2DSource            (double*,double*,void*,void*,double);

int NavierStokes2DLeftEigenvectors  (double*,double*,void*,int);
int NavierStokes2DRightEigenvectors (double*,double*,void*,int);

int NavierStokes2DUpwindRoe         (double*,double*,double*,double*,double*,double*,int,void*,double);
int NavierStokes2DUpwindRF          (double*,double*,double*,double*,double*,double*,int,void*,double);
int NavierStokes2DUpwindLLF         (double*,double*,double*,double*,double*,double*,int,void*,double);
int NavierStokes2DUpwindRusanov     (double*,double*,double*,double*,double*,double*,int,void*,double);

int NavierStokes2DWriteChem (void*,void*,double);
int NavierStokes2DPreStep (double*,void*,void*,double,double);

/*! Initialize the 2D Navier-Stokes (#NavierStokes2D) module:
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

    Keyword name       | Type         | Variable                        | Default value
    ------------------ | ------------ | ------------------------------- | ------------------------
    gamma              | double       | #NavierStokes2D::gamma          | 1.4
    Pr                 | double       | #NavierStokes2D::Pr             | 0.72
    Re                 | double       | #NavierStokes2D::Re             | -1
    T_ref              | double       | #NavierStokes2D::Tref           | 273.15 (Kelvin)
    T_0                | double       | #NavierStokes2D::T0             | 275.0  (Kelvin)
    T_S                | double       | #NavierStokes2D::TS             | 110.4  (Kelvin)
    T_A                | double       | #NavierStokes2D::TA             | 245.4  (Kelvin)
    T_B                | double       | #NavierStokes2D::TB             |  27.6  (Kelvin)
    upwinding          | char[]       | #NavierStokes2D::upw_choice     | "roe" (#_ROE_)
    include_chemistry  | char[]       | #NavierStokes2D::include_chem   | "no"

    \b Note: "physics.inp" is \b optional; if absent, default values will be used.
*/
int NavierStokes2DInitialize( void *s, /*!< Solver object of type #HyPar */
                              void *m  /*!< MPI object of type #MPIVariables */ )
{
  HyPar           *solver  = (HyPar*)          s;
  MPIVariables    *mpi     = (MPIVariables*)   m;
  NavierStokes2D  *physics = (NavierStokes2D*) solver->physics;
  int             ferr     = 0;

  static int count = 0;

  if (solver->ndims != _MODEL_NDIMS_) {
    fprintf(stderr,"Error in NavierStokes2DInitialize(): ndims has to be %d.\n",_MODEL_NDIMS_);
    return(1);
  }

  /* default values */
  physics->gamma  = 1.4;
  physics->Pr     = 0.72;
  physics->Re     = -1;
  physics->Tref   = 273.15;
  physics->T0     = 275.0;
  physics->TS     = 110.4;
  physics->TA     = 245.4;
  physics->TB     = 27.6;
  strcpy(physics->upw_choice,"roe");
  char include_chem[_MAX_STRING_SIZE_] = "no";

  /* reading physical model specific inputs - all processes */
  if (!mpi->rank) {
    FILE *in;
    if (!count) printf("Reading physical model inputs from file \"physics.inp\".\n");
    in = fopen("physics.inp","r");
    if (!in) printf("Warning: File \"physics.inp\" not found. Using default values.\n");
    else {
      char word[_MAX_STRING_SIZE_];
      ferr = fscanf(in,"%s",word);                      if (ferr != 1) return(1);
      if (!strcmp(word, "begin")){
        while (strcmp(word, "end")){
          ferr = fscanf(in,"%s",word);                  if (ferr != 1) return(1);
          if (!strcmp(word, "gamma")) {
            ferr = fscanf(in,"%lf",&physics->gamma);    if (ferr != 1) return(1);
          } else if (!strcmp(word,"upwinding")) {
            ferr = fscanf(in,"%s",physics->upw_choice); if (ferr != 1) return(1);
          } else if (!strcmp(word,"Pr")) {
            ferr = fscanf(in,"%lf",&physics->Pr);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"Re")) {
            ferr = fscanf(in,"%lf",&physics->Re);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"T_ref")) {
            ferr = fscanf(in,"%lf",&physics->Tref);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"T_0")) {
            ferr = fscanf(in,"%lf",&physics->T0);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"T_S")) {
            ferr = fscanf(in,"%lf",&physics->TS);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"T_A")) {
            ferr = fscanf(in,"%lf",&physics->TA);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"T_B")) {
            ferr = fscanf(in,"%lf",&physics->TB);       if (ferr != 1) return(1);
          } else if (!strcmp(word,"include_chemistry")) {
            ferr = fscanf(in,"%s",include_chem);        if (ferr != 1) return(1);
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
    }
    fclose(in);

    physics->include_chem = (!strcmp(include_chem,"yes"));
  }

  MPIBroadcast_character (physics->upw_choice   ,_MAX_STRING_SIZE_,0,&mpi->world);
  MPIBroadcast_double    (&physics->gamma       ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->Pr          ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->Re          ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->Tref        ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->T0          ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->TS          ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->TA          ,1                ,0,&mpi->world);
  MPIBroadcast_double    (&physics->TB          ,1                ,0,&mpi->world);
  MPIBroadcast_integer   (&physics->include_chem,1                ,0,&mpi->world);

  /* initializing physical model-specific functions */
  solver->PreStep               = NavierStokes2DPreStep;
  solver->ComputeCFL            = NavierStokes2DComputeCFL;
  solver->FFunction             = NavierStokes2DFlux;
  solver->SFunction             = NavierStokes2DSource;
  solver->AveragingFunction     = NavierStokes2DRoeAverage;
  solver->GetLeftEigenvectors   = NavierStokes2DLeftEigenvectors;
  solver->GetRightEigenvectors  = NavierStokes2DRightEigenvectors;
  solver->ParabolicFunction     = NavierStokes2DParabolicFunction;
  if      (!strcmp(physics->upw_choice,_ROE_    )) solver->Upwind = NavierStokes2DUpwindRoe;
  else if (!strcmp(physics->upw_choice,_RF_     )) solver->Upwind = NavierStokes2DUpwindRF;
  else if (!strcmp(physics->upw_choice,_LLF_    )) solver->Upwind = NavierStokes2DUpwindLLF;
  else if (!strcmp(physics->upw_choice,_RUSANOV_)) solver->Upwind = NavierStokes2DUpwindRusanov;
  else {
    if (!mpi->rank) {
      fprintf(stderr,"Error in NavierStokes2DInitialize(): %s is not a valid upwinding scheme. ",
              physics->upw_choice);
      fprintf(stderr,"Choices are %s, %s, %s, and %s.\n",_ROE_,_RF_,_LLF_,_RUSANOV_);
    }
    return(1);
  }

  /* set the value of gamma in all the boundary objects */
  int n;
  DomainBoundary  *boundary = (DomainBoundary*) solver->boundary;
  for (n = 0; n < solver->nBoundaryZones; n++)  boundary[n].gamma = physics->gamma;

  physics->nvars = _NS2D_NVARS_;

  if (physics->include_chem) {
    solver->PhysicsOutput = NavierStokes2DWriteChem;
    physics->chem = (Chemistry*) calloc (1, sizeof(Chemistry));
    ChemistryInitialize( solver,
                         physics->chem,
                         mpi );
    Chemistry* chem = (Chemistry*) physics->chem;
    physics->gamma = chem->gamma;
    physics->Tref = chem->Ti;
    physics->nvars += chem->n_reacting_species;

    if (physics->Re > 0) {
      if (!mpi->rank) {
        printf("NavierStokes2DInitialize(): Computing Reynolds and Prandtl's number based on photochemistry setup; ignoring values in physics.inp.\n");
      }
      double mu_ref = chem->mu0 * raiseto(chem->Ti/physics->T0, 1.5) * ((physics->T0+physics->TS)/(chem->Ti+physics->TS));
      double kappa_ref =  chem->kappa0
                        * raiseto(chem->Ti/physics->T0, 1.5)
                        * (  (physics->T0+physics->TA*exp(-physics->TB/physics->T0))
                           / (chem->Ti+physics->TA*exp(-physics->TB/chem->Ti))    );
      physics->Re = chem->rho_ref * chem->v_ref * chem->L_ref / mu_ref;
      physics->Pr = chem->gamma*chem->R/(chem->gamma-1) * mu_ref / kappa_ref;
    }
  }

  if (solver->nvars != physics->nvars) {
    fprintf(stderr,"Error in NavierStokes2DInitialize(): nvars has to be %d in solver.inp.\n",physics->nvars);
    return(1);
  }

  if (!mpi->rank) {
    printf("NavierStokes2D parameters:\n");
    printf("    gamma: %1.4e\n", physics->gamma);
    printf("    Reynolds number: %1.4e\n", physics->Re);
    printf("    Prandtl number: %1.4e\n", physics->Pr);
    printf("    Reference temperature: %1.4e [K]\n", physics->Tref);
    printf("    upwinding scheme: %s\n", physics->upw_choice);
    printf("    include chemistry: %s\n", (physics->include_chem?"yes":"no"));
  }

  count++;
  return(0);
}
