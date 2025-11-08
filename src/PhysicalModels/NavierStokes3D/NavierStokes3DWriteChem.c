/*! @file NavierStokes3DWriteChem.c
    @author Debojyoti Ghosh
    @brief Function to write out the reaction-related variables
*/
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <common.h>
#include <arrayfunctions.h>
#include <io.h>
#include <mpivars.h>
#include <hypar.h>
#include <physicalmodels/navierstokes3d.h>

/*! Write out the reacting species data to file */
int NavierStokes3DWriteChem( void*   s,  /*!< Solver object of type #HyPar */
                             void*   m,  /*!< MPI object of type #MPIVariables */
                             double  a_t /*!< Current simulation time */ )
{
  HyPar* solver = (HyPar*) s;
  MPIVariables* mpi = (MPIVariables*) m;
  NavierStokes3D* params = (NavierStokes3D*) solver->physics;

  if (params->include_chem) ChemistryWriteSpecies(solver, solver->u, params->chem, mpi, a_t);

  return 0;
}
