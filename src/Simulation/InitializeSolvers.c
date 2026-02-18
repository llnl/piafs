// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file InitializeSolvers.c
    @author Debojyoti Ghosh
    @brief Initialize all solvers
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <tridiagLU.h>
#include <timeintegration.h>
#include <interpolation.h>
#include <firstderivative.h>
#include <mpivars.h>
#include <simulation_object.h>

/* Function declarations */
int  ApplyBoundaryConditions     (void*,void*,double*,double*,double);
int  HyperbolicFunction          (double*,double*,void*,void*,double,int,
                                  int(*)(double*,double*,int,void*,double),
                                  int(*)(double*,double*,double*,double*,double*,
                                         double*,int,void*,double));
int  SourceFunction              (double*,double*,void*,void*,double);
int  VolumeIntegral              (double*,double*,void*,void*);
int  BoundaryIntegral            (void*,void*);
int  CalculateConservationError  (void*,void*);
void IncrementFilenameIndex      (char*,int);
int  NonLinearInterpolation      (double*,void*,void*,double,
                                  int(*)(double*,double*,int,void*,double));

/*! This function initializes all solvers-specific function pointers
    depending on user input. The specific functions used for spatial
    discretization, time integration, and solution output are set here.
*/
int InitializeSolvers(  void  *s,   /*!< Array of simulation objects of type #SimulationObject */
                        int   nsims /*!< number of simulation objects */
                     )
{
  SimulationObject *sim = (SimulationObject*) s;
  int ns;
  _DECLARE_IERR_;

  if (nsims == 0) return 0;

  if (!sim[0].mpi.rank) {
    printf("Initializing solvers.\n");
  }

  for (ns = 0; ns < nsims; ns++) {

    HyPar           *solver   = &(sim[ns].solver);
    MPIVariables    *mpi      = &(sim[ns].mpi);

    solver->ApplyBoundaryConditions = ApplyBoundaryConditions;
    solver->SourceFunction = SourceFunction;
    solver->HyperbolicFunction = HyperbolicFunction;
    solver->VolumeIntegralFunction      = VolumeIntegral;
    solver->BoundaryIntegralFunction    = BoundaryIntegral;
    solver->CalculateConservationError  = CalculateConservationError;
    solver->NonlinearInterp             = NonLinearInterpolation;

    /* choose the type of parabolic discretization */
    solver->ParabolicFunction         = NULL;
    solver->FirstDerivativePar        = NULL;

    if (!strcmp(solver->spatial_scheme_par,_SECOND_ORDER_CENTRAL_)) {
      solver->FirstDerivativePar = FirstDerivativeFirstOrder;
    } else if (!strcmp(solver->spatial_scheme_par,_FOURTH_ORDER_CENTRAL_)) {
      solver->FirstDerivativePar = FirstDerivativeFourthOrderCentral;
    } else {
      fprintf(stderr,"Error (domain %d): %s is not a supported ",
              ns, solver->spatial_scheme_par);
      fprintf(stderr,"spatial scheme for the parabolic terms.\n");
    }

    /* Spatial interpolation for hyperbolic term */
    solver->interp                = NULL;
    solver->compact               = NULL;
    solver->lusolver              = NULL;
    solver->SetInterpLimiterVar   = NULL;
    solver->flag_nonlinearinterp  = 1;
    if (strcmp(solver->interp_type,_CHARACTERISTIC_) && strcmp(solver->interp_type,_COMPONENTS_)) {
      fprintf(stderr,"Error in InitializeSolvers() (domain %d): %s is not a ",
              ns, solver->interp_type);
      fprintf(stderr,"supported interpolation type.\n");
      return(1);
    }

    if (!strcmp(solver->spatial_scheme_hyp,_FIRST_ORDER_UPWIND_)) {

      /* First order upwind scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimFirstOrderUpwindChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimFirstOrderUpwind;
      }

    } else if (!strcmp(solver->spatial_scheme_hyp,_SECOND_ORDER_CENTRAL_)) {

      /* Second order central scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimSecondOrderCentralChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimSecondOrderCentral;
      }

    } else if (!strcmp(solver->spatial_scheme_hyp,_SECOND_ORDER_MUSCL_)) {

      /* Second order MUSCL scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimSecondOrderMUSCLChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimSecondOrderMUSCL;
      }
      solver->interp = (MUSCLParameters*) calloc(1,sizeof(MUSCLParameters));
      IERR MUSCLInitialize(solver,mpi); CHECKERR(ierr);

    } else if (!strcmp(solver->spatial_scheme_hyp,_THIRD_ORDER_MUSCL_)) {

      /* Third order MUSCL scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimThirdOrderMUSCLChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimThirdOrderMUSCL;
      }
      solver->interp = (MUSCLParameters*) calloc(1,sizeof(MUSCLParameters));
      IERR MUSCLInitialize(solver,mpi); CHECKERR(ierr);

    } else if (!strcmp(solver->spatial_scheme_hyp,_FOURTH_ORDER_CENTRAL_)) {

      /* Fourth order central scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimFourthOrderCentralChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimFourthOrderCentral;
      }

    } else if (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_UPWIND_)) {

      /* Fifth order upwind scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderUpwindChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderUpwind;
      }

    } else if (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_COMPACT_UPWIND_)) {

      /* Fifth order compact upwind scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderCompactUpwindChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderCompactUpwind;
      }
      solver->compact = (CompactScheme*) calloc(1,sizeof(CompactScheme));
      IERR CompactSchemeInitialize(solver,mpi,solver->interp_type);
      solver->lusolver = (TridiagLU*) calloc (1,sizeof(TridiagLU));
      IERR tridiagLUInit(solver->lusolver,&mpi->world);CHECKERR(ierr);

    } else if (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_WENO_)) {

      /* Fifth order WENO scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderWENOChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderWENO;
      }
      solver->interp = (WENOParameters*) calloc(1,sizeof(WENOParameters));
      IERR WENOInitialize(solver,mpi,solver->spatial_scheme_hyp,solver->interp_type); CHECKERR(ierr);
      solver->flag_nonlinearinterp = !(((WENOParameters*)solver->interp)->no_limiting);

    } else if (!strcmp(solver->spatial_scheme_hyp,_FIFTH_ORDER_CRWENO_)) {

      /* Fifth order CRWENO scheme */
      if ((solver->nvars > 1) && (!strcmp(solver->interp_type,_CHARACTERISTIC_))) {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderCRWENOChar;
      } else {
        solver->InterpolateInterfacesHyp = Interp1PrimFifthOrderCRWENO;
      }
      solver->interp = (WENOParameters*) calloc(1,sizeof(WENOParameters));
      IERR WENOInitialize(solver,mpi,solver->spatial_scheme_hyp,solver->interp_type); CHECKERR(ierr);
      solver->flag_nonlinearinterp = !(((WENOParameters*)solver->interp)->no_limiting);
      solver->compact = (CompactScheme*) calloc(1,sizeof(CompactScheme));
      IERR CompactSchemeInitialize(solver,mpi,solver->interp_type);
      solver->lusolver = (TridiagLU*) calloc (1,sizeof(TridiagLU));
      IERR tridiagLUInit(solver->lusolver,&mpi->world);CHECKERR(ierr);

    } else {

      fprintf(stderr,"Error (domain %d): %s is a not a supported spatial interpolation scheme.\n",
              ns, solver->spatial_scheme_hyp);
      return(1);
    }

    /* Time integration */
    solver->time_integrator = NULL;
    if (!strcmp(solver->time_scheme,_FORWARD_EULER_)) {
      solver->TimeIntegrate = TimeForwardEuler;
      solver->msti = NULL;
    } else if (!strcmp(solver->time_scheme,_RK_)) {
      solver->TimeIntegrate = TimeRK;
      solver->msti = (ExplicitRKParameters*) calloc (1,sizeof(ExplicitRKParameters));
      IERR TimeExplicitRKInitialize(solver->time_scheme,solver->time_scheme_type,
                                    solver->msti,mpi); CHECKERR(ierr);
    } else {
      fprintf(stderr,"Error (domain %d): %s is a not a supported time-integration scheme.\n",
              ns, solver->time_scheme);
      return(1);
    }

    /* Solution output function */
    solver->WriteOutput    = NULL; /* default - no output */
    solver->filename_index = NULL;
    strcpy(solver->op_fname_root, "op");
    strcpy(solver->aux_op_fname_root, "ts0");
    if (!strcmp(solver->output_mode,"serial")) {
      solver->index_length = 5;
      solver->filename_index = (char*) calloc (solver->index_length+1,sizeof(char));
      int i; for (i=0; i<solver->index_length; i++) solver->filename_index[i] = '0';
      solver->filename_index[solver->index_length] = (char) 0;
      if (!strcmp(solver->op_file_format,"text")) {
        solver->WriteOutput = WriteText;
        strcpy(solver->solnfilename_extn,".dat");
      } else if (!strcmp(solver->op_file_format,"tecplot2d")) {
        solver->WriteOutput = WriteTecplot2D;
        strcpy(solver->solnfilename_extn,".dat");
      } else if ((!strcmp(solver->op_file_format,"binary")) || (!strcmp(solver->op_file_format,"bin"))) {
        solver->WriteOutput = WriteBinary;
        strcpy(solver->solnfilename_extn,".bin");
      } else if (!strcmp(solver->op_file_format,"none")) {
        solver->WriteOutput = NULL;
      } else {
        fprintf(stderr,"Error (domain %d): %s is not a supported file format.\n",
                ns, solver->op_file_format);
        return(1);
      }
      if ((!strcmp(solver->op_overwrite,"no")) && solver->restart_iter) {
        /* if it's a restart run, fast-forward the filename */
        int t;
        for (t=0; t<solver->restart_iter; t++)
          if ((t+1)%solver->file_op_iter == 0) IncrementFilenameIndex(solver->filename_index,solver->index_length);
      }
    } else if (!strcmp(solver->output_mode,"parallel")) {
      if (!strcmp(solver->op_file_format,"none")) solver->WriteOutput = NULL;
      else {
        /* only binary file writing supported in parallel mode */
        /* use post-processing scripts to convert              */
        solver->WriteOutput = WriteBinary;
        strcpy(solver->solnfilename_extn,".bin");
      }
    } else {
      fprintf(stderr,"Error (domain %d): %s is not a supported output mode.\n",
              ns, solver->output_mode);
      fprintf(stderr,"Should be \"serial\" or \"parallel\".    \n");
      return(1);
    }

  }

  return(0);
}
