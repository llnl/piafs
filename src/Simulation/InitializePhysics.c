/*! @file InitializePhysics.c
    @author Debojyoti Ghosh
    @brief Initialize the physical model
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <interpolation.h>
#include <mpivars.h>
#include <simulation_object.h>

/* include header files for each physical model */
#include <physicalmodels/euler1d.h>
#include <physicalmodels/navierstokes2d.h>
#include <physicalmodels/navierstokes3d.h>

/*! Initialize the physical model for a simulation: Depending on the
    physical model specified, this function calls the initialization
    function for that physical model. The latter is responsible for
    setting all the physics-specific functions that are required
    by the model.
*/
int InitializePhysics(  void  *s,   /*!< Array of simulation objects of type #SimulationObject */
                        int   nsims /*!< number of simulation objects */
                     )
{
  SimulationObject *sim = (SimulationObject*) s;
  int ns;
  _DECLARE_IERR_;

  if (nsims == 0) return 0;

  if (!sim[0].mpi.rank) {
    printf("Initializing physics. Model = \"%s\"\n",sim[0].solver.model);
  }

  for (ns = 0; ns < nsims; ns++) {

    HyPar        *solver   = &(sim[ns].solver);
    MPIVariables *mpi      = &(sim[ns].mpi);

    /* Initialize physics-specific functions to NULL */
    solver->ComputeCFL            = NULL;
    solver->FFunction             = NULL;
    solver->SFunction             = NULL;
    solver->Upwind                = NULL;
    solver->PostStage             = NULL;
    solver->PreStep               = NULL;
    solver->PostStep              = NULL;
    solver->PrintStep             = NULL;
    solver->PhysicsOutput         = NULL;
    solver->PhysicsInput          = NULL;
    solver->AveragingFunction     = NULL;
    solver->GetLeftEigenvectors   = NULL;
    solver->GetRightEigenvectors  = NULL;

    if (!strcmp(solver->model,_EULER_1D_)) {

      solver->physics = (Euler1D*) calloc (1,sizeof(Euler1D));
      IERR Euler1DInitialize(solver,mpi); CHECKERR(ierr);

    } else if (!strcmp(solver->model,_NAVIER_STOKES_2D_)) {

      solver->physics = (NavierStokes2D*) calloc (1,sizeof(NavierStokes2D));
      IERR NavierStokes2DInitialize(solver,mpi); CHECKERR(ierr);

    } else if (!strcmp(solver->model,_NAVIER_STOKES_3D_)) {

      solver->physics = (NavierStokes3D*) calloc (1,sizeof(NavierStokes3D));
      IERR NavierStokes3DInitialize(solver,mpi); CHECKERR(ierr);

    }else {

      fprintf(stderr,"Error (domain %d): %s is not a supported physical model.\n",
              ns, solver->model);
      return(1);

    }

    /* some checks */
    if ( ( (solver->GetLeftEigenvectors == NULL) || (solver->GetRightEigenvectors == NULL) )
        && (!strcmp(solver->interp_type,_CHARACTERISTIC_)) && (solver->nvars > 1) ) {
      if (!mpi->rank) {
        fprintf(stderr,"Error (domain %d): Interpolation type is defined as characteristic ", ns);
        fprintf(stderr,"but physics initializations returned NULL pointers for ");
        fprintf(stderr,"Get(Left,Right)Eigenvectors needed for characteristic-based ");
        fprintf(stderr,"reconstruction.\n");
      }
      return(1);
    }

  }

  return(0);
}
