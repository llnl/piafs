/*! @file ReadInputs.c
    @author Debojyoti Ghosh
    @brief Read the input parameters from \b solver.inp
*/

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basic.h>
#include <arrayfunctions.h>
#include <timeintegration.h>
#include <mpivars.h>
#include <simulation_object.h>

/*! Compute the load-balanced processor distribution for a given grid size
 * and the total number of MPI ranks available */
static int ComputeProcessorDistribution(int* const       a_iprocs, /*!< Processor distribution to compute */
                                        const int        a_ndims, /*!< Number of spatial dimensions */
                                        const int        a_nproc, /*!< Number of MPI ranks */
                                        const int* const a_dim     /*!< Grid dimensions */)
{
  int i;

  /* get the normal vector for the grid dimensions */
  double dvec[a_ndims]; _ArraySetValue_(dvec, a_ndims, 0.0);
  double magn = 0.0;
  for (i = 0; i < a_ndims; i++) { magn += (double) (a_dim[i]*a_dim[i]); }
  magn = sqrt(magn);
  if (magn > _MACHINE_ZERO_) {
    _ArrayCopy1D_( a_dim, dvec, a_ndims );
    _ArrayScale1D_( dvec, (1.0/magn), a_ndims );
  }

  /* calculate the maximum number of MPI ranks in each dimension */
  int max_procs[a_ndims], min_procs[a_ndims];
  for (i = 0; i < a_ndims; i++) {
    max_procs[i] = a_dim[i]/_MIN_GRID_PTS_PER_PROC_;
    min_procs[i] = 1;
  }
  int max_nproc; _ArrayProduct1D_(max_procs, a_ndims, max_nproc);
  if (max_nproc < a_nproc) {
    fprintf(stderr, "Error in ComputeProcessorDistribution():\n");
    fprintf(stderr, "  Number of MPI ranks greater than the maximum number of MPI ranks that can be used.\n");
    fprintf(stderr, "  Please re-run with %d MPI ranks.\n", max_nproc);
    return 1;
  }

  /* find all the processor distributions that are okay, i.e., their product
   * is the total number of MPI rank */
  int** iproc_candidates = NULL;
  int num_iproc_candidates = 0;
  {
    int iproc[a_ndims], ubound[a_ndims], done = 0, d;
    for (d = 0; d < a_ndims; d++) ubound[d] = max_procs[d]+1;
    _ArraySetValue_(iproc, a_ndims, 1);
    while (!done) {
      int prod; _ArrayProduct1D_(iproc, a_ndims, prod);
      if (prod == a_nproc) { num_iproc_candidates++; }
      _ArrayIncrementIndexWithLBound_(a_ndims,ubound,min_procs,iproc,done);
    }
  }
  if (num_iproc_candidates == 0) {
    fprintf(stderr, "Error in ComputeProcessorDistribution():\n");
    fprintf(stderr, "  Unable to fine candidate iprocs!\n");
    return 1;
  }
  iproc_candidates = (int**) calloc(num_iproc_candidates, sizeof(int*));
  {
    int iproc[a_ndims], ubound[a_ndims], done = 0, d, ic = 0;
    for (d = 0; d < a_ndims; d++) ubound[d] = max_procs[d]+1;
    _ArraySetValue_(iproc, a_ndims, 1);
    while (!done) {
      int prod; _ArrayProduct1D_(iproc, a_ndims, prod);
      if (prod == a_nproc) {
        iproc_candidates[ic] = (int*) calloc( a_ndims, sizeof(int));
        for (int d = 0; d < a_ndims; d++) iproc_candidates[ic][d] = iproc[d];
        ic++;
      }
      _ArrayIncrementIndexWithLBound_(a_ndims,ubound,min_procs,iproc,done);
    }
    if (ic != num_iproc_candidates) {
      fprintf(stderr, "Error in ComputeProcessorDistribution() - something went wrong!\n");
      return 1;
    }
  }

  /* find the candidate that is closest to the normalized dim vector */
  double min_norm = DBL_MAX;
  for (int i = 0; i < num_iproc_candidates; i++) {
    int d;
    double pvec[a_ndims]; _ArraySetValue_(pvec, a_ndims, 0.0);
    double magn = 0.0;
    for (d = 0; d < a_ndims; d++) { magn += (double) (iproc_candidates[i][d]*iproc_candidates[i][d]); }
    magn = sqrt(magn);
    if (magn > _MACHINE_ZERO_) {
      _ArrayCopy1D_( iproc_candidates[i], pvec, a_ndims );
      _ArrayScale1D_(pvec, (1.0/magn), a_ndims);
    }
    double norm = 0.0;
    for (d = 0; d < a_ndims; d++) {
      double tmp = pvec[d] - dvec[d];
      norm += (tmp * tmp);
    }
    if (norm < min_norm) {
      min_norm = norm;
      _ArrayCopy1D_(iproc_candidates[i], a_iprocs, a_ndims);
    }
  }

  return 0;
}

/*! Read the simulation inputs from the file \b solver.inp.
    Rank 0 reads in the inputs and broadcasts them to all the
    processors.\n\n
    The format of \b solver.inp is as follows:\n

        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

    where the list of keywords and their type are:\n
    Keyword name       | Type         | Variable                      | Default value
    ------------------ | ------------ | ----------------------------- | ------------------------------------------
    ndims              | int          | #HyPar::ndims                 | 1
    nvars              | int          | #HyPar::nvars                 | 1
    size               | int[ndims]   | #HyPar::dim_global            | must be specified
    ghost              | int          | #HyPar::ghosts                | 1
    n_iter             | int          | #HyPar::n_iter                | 0
    restart_iter       | int          | #HyPar::restart_iter          | 0
    time_scheme        | char[]       | #HyPar::time_scheme           | euler
    time_scheme_type   | char[]       | #HyPar::time_scheme_type      | none
    hyp_space_scheme   | char[]       | #HyPar::spatial_scheme_hyp    | 1
    hyp_interp_type    | char[]       | #HyPar::interp_type           | characteristic
    par_space_scheme   | char[]       | #HyPar::spatial_scheme_par    | 2
    dt                 | double       | #HyPar::dt                    | 0.0
    conservation_check | char[]       | #HyPar::ConservationCheck     | no
    screen_op_iter     | int          | #HyPar::screen_op_iter        | 1
    file_op_iter       | int          | #HyPar::file_op_iter          | 1000
    op_file_format     | char[]       | #HyPar::op_file_format        | text
    ip_file_type       | char[]       | #HyPar::ip_file_type          | ascii
    input_mode         | char[]       | #HyPar::input_mode            | serial
    output_mode        | char[]       | #HyPar::output_mode           | serial
    op_overwrite       | char[]       | #HyPar::op_overwrite          | no
    model              | char[]       | #HyPar::model                 | must be specified
    size_exact         | int[ndims]   | #HyPar::dim_global_ex         | #HyPar::dim_global

    \b Notes:
    + "ndims" \b must be specified \b before "size".
    + if "input_mode" or "output_mode" are set to "parallel" or "mpi-io",
      the number of I/O ranks must be specified right after as an integer.
      For example:

          begin
              ...
              input_mode  parallel 4
              ...
          end

      This means that 4 MPI ranks will participate in file I/O (assuming
      total MPI ranks is more than 4) (see ReadArrayParallel(),
      WriteArrayParallel(), ReadArrayMPI_IO() ).
      - The number of I/O ranks specified for "input_mode" and "output_mode"
        \b must \b be \b same. Otherwise, the value for the one specified last
        will be used.
      - The number of I/O ranks must be such that the total number of MPI ranks
        is an integer multiple. Otherwise, the code will use only 1 I/O rank.
    + If any of the keywords are not present, the default value is used, except
      the ones whose default values say "must be specified". Thus, keywords that
      are not required for a particular simulation may be left out of the
      solver.inp input file. For example,
      - a #Euler1D simulation does not need "par_space_scheme"
        because it does not have a parabolic term.
      - unless a conservation check is required, "conservation_check" can be left
        out and the code will not check for conservation.
*/
int ReadInputs( void  *s,     /*!< Array of simulation objects of type #SimulationObject
                                   of size nsims */
                int   nsims,  /*!< Number of simulation objects */
                int   rank    /*!< MPI rank of this process */
              )
{
  SimulationObject *sim = (SimulationObject*) s;
  int n, ferr    = 0;

  if (sim == NULL) {
    printf("Error: simulation object array is NULL!\n");
    printf("Please consider killing this run.\n");
    return(1);
  }

  if (!rank) {

    /* set some default values for optional inputs */
    for (n = 0; n < nsims; n++) {
      sim[n].solver.ndims           = 1;
      sim[n].solver.nvars           = 1;
      sim[n].solver.ghosts          = 1;
      sim[n].solver.dim_global      = NULL;
      sim[n].solver.dim_local       = NULL;
      sim[n].solver.dim_global_ex   = NULL;
      sim[n].mpi.iproc              = NULL;
      sim[n].mpi.N_IORanks          = 1;
      sim[n].solver.dt              = 0.0;
      sim[n].solver.n_iter          = 0;
      sim[n].solver.restart_iter    = 0;
      sim[n].solver.screen_op_iter  = 1;
      sim[n].solver.file_op_iter    = 1000;
      sim[n].solver.write_residual  = 0;
      strcpy(sim[n].solver.time_scheme        ,"euler"         );
      strcpy(sim[n].solver.time_scheme_type   ," "             );
      strcpy(sim[n].solver.spatial_scheme_hyp ,"1"             );
      strcpy(sim[n].solver.spatial_scheme_par ,"2"             );
      strcpy(sim[n].solver.interp_type        ,"characteristic");
      strcpy(sim[n].solver.ip_file_type       ,"ascii"         );
      strcpy(sim[n].solver.input_mode         ,"serial"        );
      strcpy(sim[n].solver.output_mode        ,"serial"        );
      strcpy(sim[n].solver.op_file_format     ,"text"          );
      strcpy(sim[n].solver.op_overwrite       ,"no"            );
      strcpy(sim[n].solver.model              ,"none"          );
      strcpy(sim[n].solver.ConservationCheck  ,"no"            );
    }

    /* open the file */
    FILE *in;
    printf("Reading solver inputs from file \"solver.inp\".\n");
    in = fopen("solver.inp","r");
    if (!in) {
      fprintf(stderr,"Error: File \"solver.inp\" not found.\n");
      fprintf(stderr,"Please consider killing this run.\n");
      return(1);
    }

    /* reading solver inputs */
    char word[_MAX_STRING_SIZE_];
    ferr = fscanf(in,"%s",word); if (ferr != 1) return(1);

    if (!strcmp(word, "begin")){

      while (strcmp(word, "end")) {

        ferr = fscanf(in,"%s",word); if (ferr != 1) return(1);

        if (!strcmp(word, "ndims")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.ndims)); if (ferr != 1) return(1);
          sim[0].solver.dim_global    = (int*) calloc (sim[0].solver.ndims,sizeof(int));
          sim[0].mpi.iproc            = (int*) calloc (sim[0].solver.ndims,sizeof(int));
          sim[0].solver.dim_global_ex = (int*) calloc (sim[0].solver.ndims,sizeof(int));

          int n;
          for (n = 1; n < nsims; n++) {
            sim[n].solver.ndims = sim[0].solver.ndims;
            sim[n].solver.dim_global    = (int*) calloc (sim[n].solver.ndims,sizeof(int));
            sim[n].mpi.iproc            = (int*) calloc (sim[n].solver.ndims,sizeof(int));
            sim[n].solver.dim_global_ex = (int*) calloc (sim[n].solver.ndims,sizeof(int));
          }

        } else if (!strcmp(word, "nvars")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.nvars));
          for (int n = 1; n < nsims; n++) sim[n].solver.nvars = sim[0].solver.nvars;

        } else if   (!strcmp(word, "size")) {

          for (int n = 0; n < nsims; n++) {
            if (!sim[n].solver.dim_global) {
              fprintf(stderr,"Error in ReadInputs(): dim_global not allocated for n=%d.\n", n);
              fprintf(stderr,"Please specify ndims before dimensions.\n"         );
              return(1);
            } else {
              for (int i=0; i<sim[n].solver.ndims; i++) {
                ferr = fscanf(in,"%d",&(sim[n].solver.dim_global[i]));
                if (ferr != 1) {
                  fprintf(stderr,"Error in ReadInputs() while reading grid sizes for domain %d.\n", n);
                  return(1);
                }
                sim[n].solver.dim_global_ex[i] = sim[n].solver.dim_global[i];
              }
            }
          }

        } else if   (!strcmp(word, "size_exact")) {

          for (int n = 0; n < nsims; n++) {
            if (!sim[n].solver.dim_global_ex) {
              fprintf(stderr,"Error in ReadInputs(): dim_global_ex not allocated for n=%d.\n", n);
              fprintf(stderr,"Please specify ndims before dimensions.\n"         );
              return(1);
            } else {
              for (int i=0; i<sim[n].solver.ndims; i++) {
                ferr = fscanf(in,"%d",&(sim[n].solver.dim_global_ex[i]));
                if (ferr != 1) {
                  fprintf(stderr,"Error in ReadInputs() while reading exact solution grid sizes for domain %d.\n", n);
                  return(1);
                }
              }
            }
          }

        } else if (!strcmp(word, "ghost")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.ghosts));

          int n;
          for (n = 1; n < nsims; n++) sim[n].solver.ghosts = sim[0].solver.ghosts;

        } else if (!strcmp(word, "n_iter")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.n_iter));

          int n;
          for (n = 1; n < nsims; n++) sim[n].solver.n_iter = sim[0].solver.n_iter;

        } else if (!strcmp(word, "restart_iter")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.restart_iter));

          int n;
          for (n = 1; n < nsims; n++) sim[n].solver.restart_iter = sim[0].solver.restart_iter;

        } else if (!strcmp(word, "time_scheme")) {

          ferr = fscanf(in,"%s",sim[0].solver.time_scheme);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.time_scheme, sim[0].solver.time_scheme);

        }  else if (!strcmp(word, "time_scheme_type" )) {

          ferr = fscanf(in,"%s",sim[0].solver.time_scheme_type);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.time_scheme_type, sim[0].solver.time_scheme_type);

        }  else if (!strcmp(word, "hyp_space_scheme")) {

          ferr = fscanf(in,"%s",sim[0].solver.spatial_scheme_hyp);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.spatial_scheme_hyp, sim[0].solver.spatial_scheme_hyp);

        }  else if (!strcmp(word, "hyp_interp_type")) {

          ferr = fscanf(in,"%s",sim[0].solver.interp_type);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.interp_type, sim[0].solver.interp_type);

        }  else if (!strcmp(word, "par_space_scheme")) {

          ferr = fscanf(in,"%s",sim[0].solver.spatial_scheme_par);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.spatial_scheme_par, sim[0].solver.spatial_scheme_par);

        }  else if (!strcmp(word, "dt")) {

          ferr = fscanf(in,"%lf",&(sim[0].solver.dt));

          int n;
          for (n = 1; n < nsims; n++) sim[n].solver.dt = sim[0].solver.dt;

        }  else if (!strcmp(word, "conservation_check" )) {

          ferr = fscanf(in,"%s",sim[0].solver.ConservationCheck);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.ConservationCheck, sim[0].solver.ConservationCheck);

        }  else if (!strcmp(word, "screen_op_iter")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.screen_op_iter));

          int n;
          for (n = 1; n < nsims; n++) sim[n].solver.screen_op_iter = sim[0].solver.screen_op_iter;

        }  else if (!strcmp(word, "file_op_iter")) {

          ferr = fscanf(in,"%d",&(sim[0].solver.file_op_iter));

          int n;
          for (n = 1; n < nsims; n++) sim[n].solver.file_op_iter = sim[0].solver.file_op_iter;

        }  else if (!strcmp(word, "op_file_format")) {

          ferr = fscanf(in,"%s",sim[0].solver.op_file_format);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.op_file_format, sim[0].solver.op_file_format);

        }  else if (!strcmp(word, "ip_file_type")) {

          ferr = fscanf(in,"%s",sim[0].solver.ip_file_type);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.ip_file_type, sim[0].solver.ip_file_type);

        }  else if (!strcmp(word, "input_mode")) {

          ferr = fscanf(in,"%s",sim[0].solver.input_mode);
          if (strcmp(sim[0].solver.input_mode,"serial")) ferr = fscanf(in,"%d",&(sim[0].mpi.N_IORanks));

          int n;
          for (n = 1; n < nsims; n++) {
            strcpy(sim[n].solver.input_mode, sim[0].solver.input_mode);
            if (strcmp(sim[n].solver.input_mode,"serial")) sim[n].mpi.N_IORanks = sim[0].mpi.N_IORanks;
          }

         } else if (!strcmp(word, "output_mode"))  {

          ferr = fscanf(in,"%s",sim[0].solver.output_mode);
          if (strcmp(sim[0].solver.output_mode,"serial")) ferr = fscanf(in,"%d",&(sim[0].mpi.N_IORanks));

          int n;
          for (n = 1; n < nsims; n++) {
            strcpy(sim[n].solver.output_mode, sim[0].solver.output_mode);
            if (strcmp(sim[n].solver.output_mode,"serial")) sim[n].mpi.N_IORanks = sim[0].mpi.N_IORanks;
          }

        } else if   (!strcmp(word, "op_overwrite")) {

          ferr = fscanf(in,"%s",sim[0].solver.op_overwrite);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.op_overwrite, sim[0].solver.op_overwrite);

        }  else if (!strcmp(word, "model")) {

          ferr = fscanf(in,"%s",sim[0].solver.model);

          int n;
          for (n = 1; n < nsims; n++) strcpy(sim[n].solver.model, sim[0].solver.model);

        } else if (strcmp(word, "end")) {

          char useless[_MAX_STRING_SIZE_];
          ferr = fscanf(in,"%s",useless);
          printf("Warning: keyword %s in file \"solver.inp\" with value %s not recognized or extraneous. Ignoring.\n",
                  word,useless);

        }
        if (ferr != 1) return(1);

      }

    } else {

       fprintf(stderr,"Error: Illegal format in file \"solver.inp\".\n");
      return(1);

    }

    /* close the file */
    fclose(in);

    /* load balancing */
    for (n = 0; n < nsims; n++) {
      int err = ComputeProcessorDistribution( sim[n].mpi.iproc,
                                              sim[n].solver.ndims,
                                              sim[n].mpi.nproc,
                                              sim[n].solver.dim_global );
      if (err) return err;
    }

    /* some checks */
    for (n = 0; n < nsims; n++) {

      if (sim[n].solver.screen_op_iter <= 0)  sim[n].solver.screen_op_iter = 1;
      if (sim[n].solver.file_op_iter <= 0)    sim[n].solver.file_op_iter   = sim[n].solver.n_iter;

      /* restart only supported for binary output files */
      if ((sim[n].solver.restart_iter != 0) && strcmp(sim[n].solver.op_file_format,"binary")) {
        if (!sim[n].mpi.rank) fprintf(stderr,"Error in ReadInputs(): Restart is supported only for binary output files.\n");
        return(1);
      }
    }
  }

#ifndef serial
  for (n = 0; n < nsims; n++) {

    /* Broadcast the input parameters */
    MPIBroadcast_integer(&(sim[n].solver.ndims),1,0,&(sim[n].mpi.world));
    if (sim[n].mpi.rank) {
      sim[n].solver.dim_global    = (int*) calloc (sim[n].solver.ndims,sizeof(int));
      sim[n].mpi.iproc            = (int*) calloc (sim[n].solver.ndims,sizeof(int));
      sim[n].solver.dim_global_ex = (int*) calloc (sim[n].solver.ndims,sizeof(int));
    }
    MPIBroadcast_integer(&(sim[n].solver.nvars)         ,1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_integer( sim[n].solver.dim_global      ,sim[n].solver.ndims,0,&(sim[n].mpi.world));
    MPIBroadcast_integer( sim[n].solver.dim_global_ex   ,sim[n].solver.ndims,0,&(sim[n].mpi.world));
    MPIBroadcast_integer( sim[n].mpi.iproc              ,sim[n].solver.ndims,0,&(sim[n].mpi.world));
    MPIBroadcast_integer(&(sim[n].mpi.N_IORanks)        ,1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_integer(&(sim[n].solver.ghosts)        ,1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_integer(&(sim[n].solver.n_iter)        ,1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_integer(&(sim[n].solver.restart_iter)  ,1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_integer(&(sim[n].solver.screen_op_iter),1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_integer(&(sim[n].solver.file_op_iter)  ,1                  ,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.time_scheme        ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.time_scheme_type   ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.spatial_scheme_hyp ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.interp_type        ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.spatial_scheme_par ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.ConservationCheck  ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.op_file_format     ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.ip_file_type       ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.input_mode         ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.output_mode        ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.op_overwrite       ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));
    MPIBroadcast_character(sim[n].solver.model              ,_MAX_STRING_SIZE_,0,&(sim[n].mpi.world));

    MPIBroadcast_double(&(sim[n].solver.dt),1,0,&(sim[n].mpi.world));
  }
#endif

  return 0;
}
