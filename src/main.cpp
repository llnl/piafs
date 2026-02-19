// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file main.cpp
 *  @brief Main driver.
 * The main driver function that calls the initialization, solving, and cleaning up functions.
 *  @author Debojyoti Ghosh
*/

/*! @mainpage

  \image html piafs.jpg width=400px

  @author Debojyoti Ghosh [\b Email: ghosh5@llnl.gov],
          Albertine Oudin [\b Email: oudin1@llnl.gov]

  PIAFS: Compressible Euler/Navier-Stokes Solver on Cartesian Grids
  -------------------------------------------------------------------

  PIAFS is a finite-difference algorithm to solve the compressible Euler/Navier-Stokes
  equations (with source terms) on Cartesian grids.

  Documentation
  -------------
  To generate a local copy of this documentation, run the following in the root directory:
  \code{.sh}
  doxygen Doxyfile
  \endcode
  The folder \c doc/html will contain the generated documentation in HTML format.

  Quick Start
  -----------

  PIAFS supports two build systems: \b CMake (recommended) and \b Autotools.

  \b CMake \b build:
  \code{.sh}
  mkdir build && cd build
  cmake ..
  make -j 4
  \endcode

  \b Autotools \b build:
  \code{.sh}
  autoreconf -i
  ./configure
  make && make install
  \endcode

  For detailed build instructions, options, and troubleshooting, see:
  + \ref cmake_build "Building PIAFS with CMake" - CMake build guide
  + \ref autotools_build "Building PIAFS with Autotools" - Autotools build guide
  + \ref Input_Files - Input file documentation
  + \ref Numerical_Method - Numerical methods documentation

  \b Key \b build \b options:
  + CMake: \c -DENABLE_SERIAL=ON (serial mode), \c -DENABLE_OMP=ON (OpenMP)
  + Autotools: \c --enable-serial (serial mode), \c --enable-omp (OpenMP)

  Running
  -------
  + It's best to start with some examples. See the section on examples.
  + To run more cases, see the section on input files for a complete description of input files required.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>

#include <mpivars_cpp.h>
#include <simulation_library.h>
#include <build_info.h>
#include <gpu_runtime.h>

static const char help[] = "PIAFS - A finite-difference algorithm for the compressible Euler/Navier-Stokes equations";

/*!
 * \brief Main driver
 *
 * The main driver function that calls the initialization, solving, and cleaning up functions.
*/
int main(int argc, char **argv)
{
  /* Set stdout to line-buffered mode for better output in multi-process context */
  setvbuf(stdout, NULL, _IOLBF, 0);

  int               ierr = 0, d, n;
  struct timeval    main_start, solve_start;
  struct timeval    main_end  , solve_end  ;

#ifdef serial
  int world = 0;
  int rank  = 0;
  int nproc = 1;
  printf("================================================================================\n");
  printf("PIAFS - Serial Version\n");
  printf("  Version: %s\n", PIAFS_VERSION);
  printf("  Git Hash: %s (branch: %s)%s\n", PIAFS_GIT_HASH, PIAFS_GIT_BRANCH,
         (strcmp(PIAFS_GIT_DIRTY, "yes") == 0) ? " [dirty]" : "");
  printf("  Build Date: %s\n", PIAFS_BUILD_DATE);
  printf("  Build Type: %s\n", PIAFS_BUILD_TYPE);
  printf("  MPI Mode: %s\n", PIAFS_MPI_MODE);
  printf("  OpenMP: %s\n", PIAFS_OPENMP);
#ifdef GPU_CUDA
  {
    int gpu_count = GPUGetDeviceCount();
    if (gpu_count > 0) {
      printf("  GPU Devices: %d\n", gpu_count);
    }
  }
#elif defined(GPU_HIP)
  {
    int gpu_count = GPUGetDeviceCount();
    if (gpu_count > 0) {
      printf("  GPU Devices: %d\n", gpu_count);
    }
  }
#endif
  printf("================================================================================\n");
#else
  MPI_Comm world;
  int rank, nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank );
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  /* Compute local rank (rank within the node) for GPU assignment */
  int local_rank = 0;
  int local_size = 1;
  int gpu_count_local = 0;
  int gpu_count_total = 0;
  int num_nodes = 1;
#if defined(GPU_CUDA) || defined(GPU_HIP)
  {
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);
    MPI_Comm_free(&shmcomm);

    /* Initialize GPU for this MPI rank and get device count */
    if (GPUShouldUse()) {
      gpu_count_local = GPUInitializeMPI(rank, local_rank, local_size);
      if (gpu_count_local < 0) {
        fprintf(stderr, "[Rank %d] Error: GPU initialization failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    } else {
      gpu_count_local = GPUGetDeviceCount();
    }

    /* Count total GPUs across all nodes (only local rank 0 contributes to avoid double counting) */
    int my_gpu_contribution = (local_rank == 0) ? gpu_count_local : 0;
    int my_node_contribution = (local_rank == 0) ? 1 : 0;
    MPI_Reduce(&my_gpu_contribution, &gpu_count_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_node_contribution, &num_nodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  if (!rank) {
    printf("================================================================================\n");
    printf("PIAFS - Parallel (MPI) version with %d processes\n", nproc);
    printf("  Version: %s\n", PIAFS_VERSION);
    printf("  Git Hash: %s (branch: %s)%s\n", PIAFS_GIT_HASH, PIAFS_GIT_BRANCH,
           (strcmp(PIAFS_GIT_DIRTY, "yes") == 0) ? " [dirty]" : "");
    printf("  Build Date: %s\n", PIAFS_BUILD_DATE);
    printf("  Build Type: %s\n", PIAFS_BUILD_TYPE);
    printf("  OpenMP: %s\n", PIAFS_OPENMP);
#ifdef GPU_CUDA
    {
      if (gpu_count_total > 0) {
        printf("  GPU Devices: %d total (%d per node, %d node%s)",
               gpu_count_total, gpu_count_local, num_nodes, num_nodes > 1 ? "s" : "");
        if (GPUShouldUse()) {
          printf(" (GPU ENABLED)\n");
        } else {
          printf(" (GPU DISABLED - set PIAFS_USE_GPU=1 to enable)\n");
        }
      }
    }
#elif defined(GPU_HIP)
    {
      if (gpu_count_total > 0) {
        printf("  GPU Devices: %d total (%d per node, %d node%s)",
               gpu_count_total, gpu_count_local, num_nodes, num_nodes > 1 ? "s" : "");
        if (GPUShouldUse()) {
          printf(" (GPU ENABLED)\n");
        } else {
          printf(" (GPU DISABLED - set PIAFS_USE_GPU=1 to enable)\n");
        }
      }
    }
#endif
    printf("================================================================================\n");
  }
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
