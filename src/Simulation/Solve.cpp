// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file Solve.cpp
    @author Debojyoti Ghosh
    @brief  Solve the governing equations in time
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <common_cpp.h>
#include <io_cpp.h>
#include <timeintegration_cpp.h>
#include <mpivars_cpp.h>
#include <simulation_object.h>

extern "C" int CalculateError(void*,void*); /*!< Calculate the error in the final solution */
int OutputSolution(void*,int,double);   /*!< Write solutions to file */
extern "C" void ResetFilenameIndex(char*, int); /*!< Reset filename index */

/*! This function integrates the semi-discrete ODE (obtained from discretizing the
    PDE in space) using natively implemented time integration methods. It initializes
    the time integration object, iterates the simulation for the required number of
    time steps, and calculates the errors. After the specified number of iterations,
    it writes out some information to the screen and the solution to a file.
*/
int Solve(  void  *s,     /*!< Array of simulation objects of type #SimulationObject */
            int   nsims,  /*!< number of simulation objects */
            int   rank,   /*!< MPI rank of this process */
            int   nproc   /*!< Number of MPI processes */
         )
{
  SimulationObject* sim = (SimulationObject*) s;

  /* make sure none of the simulation objects sent in the array
   * are "barebones" type */
  for (int ns = 0; ns < nsims; ns++) {
    if (sim[ns].is_barebones == 1) {
      fprintf(stderr, "Error in Solve(): simulation object %d on rank %d is barebones!\n",
              ns, rank );
      return 1;
    }
  }

  /* Define and initialize the time-integration object */
  TimeIntegration TS;
  if (!rank) printf("Setting up time integration.\n");
  TimeInitialize(sim, nsims, rank, nproc, &TS);
  double ti_runtime = 0.0;

  if (!rank) printf("Solving in time (from %d to %d iterations)\n",TS.restart_iter,TS.n_iter);
  for (TS.iter = TS.restart_iter; TS.iter < TS.n_iter; TS.iter++) {

    /* Write initial solution to file if this is the first iteration */
    if (!TS.iter) {
      for (int ns = 0; ns < nsims; ns++) {
        if (sim[ns].solver.PhysicsOutput) {
          sim[ns].solver.PhysicsOutput( &(sim[ns].solver),
                                        &(sim[ns].mpi),
                                        TS.waqt );
        }
      }
      OutputSolution(sim, nsims, TS.waqt);
    }

    /* Call pre-step function */
    TimePreStep (&TS);

    /* Step in time */
    TimeStep (&TS);

    /* Call post-step function */
    TimePostStep (&TS);

    ti_runtime += TS.iter_wctime;

    /* Print information to screen */
    TimePrintStep(&TS);

    /* Write intermediate solution to file */
    if (      ((TS.iter+1)%sim[0].solver.file_op_iter == 0)
          &&  ((TS.iter+1) < TS.n_iter) ) {
      for (int ns = 0; ns < nsims; ns++) {
        if (sim[ns].solver.PhysicsOutput) {
          sim[ns].solver.PhysicsOutput( &(sim[ns].solver),
                                        &(sim[ns].mpi),
                                        TS.waqt );
        }
      }
      OutputSolution(sim, nsims, TS.waqt);
    }

  }

  double t_final = TS.waqt;
  TimeCleanup(&TS);

  if (!rank) {
    printf( "Completed time integration (Final time: %f), total wctime: %f (seconds).\n",
            t_final, ti_runtime );
    if (nsims > 1) printf("\n");
  }

  /* calculate error if exact solution has been provided */
  for (int ns = 0; ns < nsims; ns++) {
    CalculateError(&(sim[ns].solver),
                   &(sim[ns].mpi) );
  }

  /* write a final solution file */
  for (int ns = 0; ns < nsims; ns++) {
    if (sim[ns].solver.PhysicsOutput) {
      sim[ns].solver.PhysicsOutput( &(sim[ns].solver),
                                    &(sim[ns].mpi),
                                    t_final );
    }
  }
  OutputSolution(sim, nsims, t_final);

  return 0;
}
