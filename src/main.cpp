/*! @file main.cpp
 *  @brief Main driver.
 * The main driver function that calls the initialization, solving, and cleaning up functions.
 *  @author Debojyoti Ghosh
*/

/*! @mainpage

  @author Debojyoti Ghosh [\b Email: (first name) (dot) (last name) (at) gmail (dot) com, \b Website: http://debog.github.io/]

  PIAFS: Compressible Euler/Navier-Stokes Solver on Cartesian Grids
  -------------------------------------------------------------------

  PIAFS is a finite-difference algorithm to solve the compressible Euler/Navier-Stokes
  equations (with source terms) on Cartesian grids.

  Documentation
  -------------
  To generate a local copy of this documentation, run "doxygen Doxyfile" in $(root_dir). The folder $(root_dir)/doc
  should contain the generated documentation in HTML format.

  Compiling
  ---------

  To compile PIAFS, follow these steps in the root directory:

        autoreconf -i
        [CFLAGS="..."] [CXXFLAGS="..."] ./configure [options]
        make
        make install

  CFLAGS and CXXFLAGS should include all the compiler flags.

  \b Note: Default installation target is its own directory, and thus "make install" should not require
           administrative privileges. The binary will be placed in \a bin/ subdirectory.

  The configure options can include options such as BLAS/LAPACK location, MPI directory, etc. Type "./configure --help"
  to see a full list. The options specific to PIAFS are:
  + \--enable-serial: Compile a serial version without MPI.
  + \--with-mpi-dir: Specify path where mpicc is installed, if not in standard path.
  + \--enable-omp: Enable OpenMP threads.

  Running
  -------
  + It's best to start with some examples. See the section on examples.
  + To run more cases, see the section in input files for a complete description of input files required.

*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>

#include <mpivars_cpp.h>
#include <simulation_library.h>

static const char help[] = "PIAFS - A finite-difference algorithm for the compressible Euler/Navier-Stokes equations";

/*!
 * \brief Main driver
 *
 * The main driver function that calls the initialization, solving, and cleaning up functions.
*/
int main(int argc, char **argv)
{
  int               ierr = 0, d, n;
  struct timeval    main_start, solve_start;
  struct timeval    main_end  , solve_end  ;

#ifdef serial
  int world = 0;
  int rank  = 0;
  int nproc = 1;
  printf("PIAFS - Serial Version\n");
#else
  MPI_Comm world;
  int rank, nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank );
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  if (!rank) printf("PIAFS - Parallel (MPI) version with %d processes\n",nproc);
#endif

  gettimeofday(&main_start,NULL);

  int sim_type = -1;
  Simulation *sim = NULL;

  if (!rank) {

    std::string ensemble_sim_fname(_ENSEMBLE_SIM_INP_FNAME_);
    FILE *f_ensemble_sim = fopen(ensemble_sim_fname.c_str(), "r");

    if (f_ensemble_sim) {

      sim_type = _SIM_TYPE_ENSEMBLE_;
      fclose(f_ensemble_sim);

    } else {

      sim_type = _SIM_TYPE_SINGLE_;

    }

  }

#ifndef serial
  MPI_Bcast(&sim_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  if (sim_type == _SIM_TYPE_SINGLE_) {
    sim = new SingleSimulation;
  } else if (sim_type == _SIM_TYPE_ENSEMBLE_) {
    if (!rank) printf("-- Ensemble Simulation --\n");
    sim = new EnsembleSimulation;
  } else {
    fprintf(stderr, "ERROR: invalid sim_type (%d) on rank %d.\n",
            sim_type, rank);
  }

  if (sim == NULL) {
    fprintf(stderr, "ERROR: unable to create sim on rank %d.\n",
            rank );
    return 1;
  }

  /* Allocate simulation objects */
  ierr = sim->define(rank, nproc);
  if (!sim->isDefined()) {
    printf("Error: Simulation::define() failed on rank %d\n",
           rank);
    return 1;
  }
  if (ierr) {
    printf("Error: Simulation::define() returned with status %d on process %d.\n",
            ierr, rank);
    return(ierr);
  }

#ifndef serial
  ierr = sim->mpiCommDup();
#endif

  /* Read Inputs */
  ierr = sim->ReadInputs();
  if (ierr) {
    printf("Error: Simulation::ReadInputs() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Initialize and allocate arrays */
  ierr = sim->Initialize();
  if (ierr) {
    printf("Error: Simulation::Initialize() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* read and set grid & initial solution */
  ierr = sim->InitialSolution();
  if (ierr) {
    printf("Error: Simulation::InitialSolution() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Initialize domain boundaries */
  ierr = sim->InitializeBoundaries();
  if (ierr) {
    printf("Error: Simulation::InitializeBoundaries() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Initialize solvers */
  ierr = sim->InitializeSolvers();
  if (ierr) {
    printf("Error: Simulation::InitializeSolvers() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Initialize physics */
  ierr = sim->InitializePhysics();
  if (ierr) {
    printf("Error: Simulation::InitializePhysics() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Initialize physics data */
  ierr = sim->InitializePhysicsData();
  if (ierr) {
    printf("Error: Simulation::InitializePhysicsData() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Wrap up initializations */
  ierr = sim->InitializationWrapup();
  if (ierr) {
    printf("Error: Simulation::InitializationWrapup() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }

  /* Initializations complete */

  /* Run the solver */
#ifndef serial
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  gettimeofday(&solve_start,NULL);
  ierr = sim->Solve();
  if (ierr) {
    printf("Error: Simulation::Solve() returned with status %d on process %d.\n",ierr,rank);
    return(ierr);
  }
  gettimeofday(&solve_end,NULL);
#ifndef serial
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  gettimeofday(&main_end,NULL);

  /* calculate solver and total runtimes */
  long long walltime;
  walltime = (  (main_end.tv_sec * 1000000   + main_end.tv_usec  )
              - (main_start.tv_sec * 1000000 + main_start.tv_usec));
  double main_runtime = (double) walltime / 1000000.0;
  ierr = MPIMax_double(&main_runtime,&main_runtime,1,&world); if(ierr) return(ierr);
  walltime = (  (solve_end.tv_sec * 1000000   + solve_end.tv_usec  )
              - (solve_start.tv_sec * 1000000 + solve_start.tv_usec));
  double solver_runtime = (double) walltime / 1000000.0;
  ierr = MPIMax_double(&solver_runtime,&solver_runtime,1,&world); if(ierr) return(ierr);

  /* Write errors and other data */
  sim->WriteErrors(solver_runtime, main_runtime);

  /* Cleaning up */
  delete sim;
  if (!rank) printf("Finished.\n");

#ifndef serial
  MPI_Comm_free(&world);
  MPI_Finalize();
#endif

  return(0);
}
