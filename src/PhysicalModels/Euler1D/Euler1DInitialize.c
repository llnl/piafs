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
    include_chemistry  | char[]       | #Euler1D::include_chem        | "no"

    \b Note: "physics.inp" is \b optional; if absent, default values will be used.
*/
int Euler1DInitialize(void *s, /*!< Solver object of type #HyPar */
                      void *m  /*!< Object of type #MPIVariables containing MPI-related info */ )
{
  HyPar         *solver  = (HyPar*)         s;
  MPIVariables  *mpi     = (MPIVariables*)  m;
  Euler1D       *physics = (Euler1D*)       solver->physics;
  int           ferr;

  static int count = 0;

  if (solver->ndims != _MODEL_NDIMS_) {
    fprintf(stderr,"Error in Euler1DInitialize(): ndims has to be %d.\n",_MODEL_NDIMS_);
    return(1);
  }

  /* default values */
  physics->gamma = 1.4;
  strcpy(physics->upw_choice,"roe");
  physics->include_chem = 0;
  physics->chem = NULL;
  char include_chem[_MAX_STRING_SIZE_] = "no";

  /* reading physical model specific inputs */
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

    physics->include_chem = (!strcmp(include_chem,"yes"));
  }

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
  else if (!strcmp(physics->upw_choice,_RUSANOV_)) solver->Upwind = Euler1DUpwindRusanov;
  else {
    if (!mpi->rank) fprintf(stderr,"Error in Euler1DInitialize(): %s is not a valid upwinding scheme.\n",
                            physics->upw_choice);
    return(1);
  }
  solver->AveragingFunction     = Euler1DRoeAverage;
  solver->GetLeftEigenvectors   = Euler1DLeftEigenvectors;
  solver->GetRightEigenvectors  = Euler1DRightEigenvectors;

  physics->nvars = _EU1D_NVARS_;

  if (physics->include_chem) {
    solver->PhysicsOutput = Euler1DWriteChem;
    physics->chem = (Chemistry*) calloc (1, sizeof(Chemistry));
    ChemistryInitialize( solver,
                         physics->chem,
                         mpi,
                         _EU1D_NVARS_);
    Chemistry* chem = (Chemistry*) physics->chem;
    physics->gamma = chem->gamma;
    physics->nvars += chem->n_reacting_species;
  }

  if (solver->nvars != physics->nvars) {
    fprintf(stderr,"Error in Euler1DInitialize(): nvars has to be %d in solver.inp.\n",physics->nvars);
    return(1);
  }

  if (!mpi->rank) {
    printf("Euler1D parameters:\n");
    printf("    gamma: %1.4e\n", physics->gamma);
    printf("    upwinding scheme: %s\n", physics->upw_choice);
    printf("    include chemistry: %s\n", (physics->include_chem?"yes":"no"));
  }

  count++;
  return(0);
}
