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

double NavierStokes2DComputeCFL        (void*,void*,double,double);
int    NavierStokes2DFlux              (double*,double*,int,void*,double);
int    NavierStokes2DRoeAverage        (double*,double*,double*,void*);
int    NavierStokes2DParabolicFunction (double*,double*,void*,void*,double);
int    NavierStokes2DSource            (double*,double*,void*,void*,double);

int    NavierStokes2DLeftEigenvectors  (double*,double*,void*,int);
int    NavierStokes2DRightEigenvectors (double*,double*,void*,int);

int    NavierStokes2DUpwindRoe         (double*,double*,double*,double*,double*,double*,int,void*,double);
int    NavierStokes2DUpwindRF          (double*,double*,double*,double*,double*,double*,int,void*,double);
int    NavierStokes2DUpwindLLF         (double*,double*,double*,double*,double*,double*,int,void*,double);
int    NavierStokes2DUpwindSWFS        (double*,double*,double*,double*,double*,double*,int,void*,double);
int    NavierStokes2DUpwindRusanov     (double*,double*,double*,double*,double*,double*,int,void*,double);

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

    Keyword name       | Type         | Variable                                        | Default value
    ------------------ | ------------ | ----------------------------------------------- | ------------------------
    gamma              | double       | #NavierStokes2D::gamma                          | 1.4
    Pr                 | double       | #NavierStokes2D::Pr                             | 0.72
    Re                 | double       | #NavierStokes2D::Re                             | -1
    Minf               | double       | #NavierStokes2D::Minf                           | 1.0
    R                  | double       | #NavierStokes2D::R                              | 1.0
    upwinding          | char[]       | #NavierStokes2D::upw_choice                     | "roe" (#_ROE_)

    + If "HB" (#NavierStokes2D::HB) is specified as 3, it should be followed by the the
      Brunt-Vaisala frequency (#NavierStokes2D::N_bv), i.e.

        begin
            ...
            HB      3 0.01
            ...
        end

    \b Note: "physics.inp" is \b optional; if absent, default values will be used.
*/
int NavierStokes2DInitialize(
                              void *s, /*!< Solver object of type #HyPar */
                              void *m  /*!< MPI object of type #MPIVariables */
                            )
{
  HyPar           *solver  = (HyPar*)          s;
  MPIVariables    *mpi     = (MPIVariables*)   m;
  NavierStokes2D  *physics = (NavierStokes2D*) solver->physics;
  int             ferr     = 0;

  static int count = 0;

  if (solver->nvars != _MODEL_NVARS_) {
    fprintf(stderr,"Error in NavierStokes2DInitialize(): nvars has to be %d.\n",_MODEL_NVARS_);
    return(1);
  }
  if (solver->ndims != _MODEL_NDIMS_) {
    fprintf(stderr,"Error in NavierStokes2DInitialize(): ndims has to be %d.\n",_MODEL_NDIMS_);
    return(1);
  }

  /* default values */
  physics->gamma  = 1.4;
  physics->Pr     = 0.72;
  physics->Re     = -1;
  physics->Minf   = 1.0;
  physics->C1     = 1.458e-6;
  physics->C2     = 110.4;
  physics->R      = 1.0;
  strcpy(physics->upw_choice,"roe");

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
          } else if (!strcmp(word,"Minf")) {
            ferr = fscanf(in,"%lf",&physics->Minf);     if (ferr != 1) return(1);
          } else if (!strcmp(word,"R")) {
            ferr = fscanf(in,"%lf",&physics->R);        if (ferr != 1) return(1);
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
  }

  IERR MPIBroadcast_character (physics->upw_choice,_MAX_STRING_SIZE_,0,&mpi->world); CHECKERR(ierr);
  IERR MPIBroadcast_double    (&physics->gamma    ,1                ,0,&mpi->world); CHECKERR(ierr);
  IERR MPIBroadcast_double    (&physics->Pr       ,1                ,0,&mpi->world); CHECKERR(ierr);
  IERR MPIBroadcast_double    (&physics->Re       ,1                ,0,&mpi->world); CHECKERR(ierr);
  IERR MPIBroadcast_double    (&physics->Minf     ,1                ,0,&mpi->world); CHECKERR(ierr);
  IERR MPIBroadcast_double    (&physics->R        ,1                ,0,&mpi->world); CHECKERR(ierr);

  /* Scaling the Reynolds number with the M_inf */
  physics->Re /= physics->Minf;

  /* initializing physical model-specific functions */
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
  else if (!strcmp(physics->upw_choice,_SWFS_   )) solver->Upwind = NavierStokes2DUpwindSWFS;
  else if (!strcmp(physics->upw_choice,_RUSANOV_)) solver->Upwind = NavierStokes2DUpwindRusanov;
  else {
    if (!mpi->rank) {
      fprintf(stderr,"Error in NavierStokes2DInitialize(): %s is not a valid upwinding scheme. ",
              physics->upw_choice);
      fprintf(stderr,"Choices are %s, %s, %s, %s, and %s.\n",_ROE_,_RF_,_LLF_,_SWFS_,_RUSANOV_);
    }
    return(1);
  }

  /* set the value of gamma in all the boundary objects */
  int n;
  DomainBoundary  *boundary = (DomainBoundary*) solver->boundary;
  for (n = 0; n < solver->nBoundaryZones; n++)  boundary[n].gamma = physics->gamma;


  count++;
  return(0);
}
