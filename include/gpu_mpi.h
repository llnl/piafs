/*! @file gpu_mpi.h
    @brief GPU-aware MPI function declarations
*/

#ifndef _GPU_MPI_H_
#define _GPU_MPI_H_

#ifdef __cplusplus
extern "C" {
#endif

/* GPU-aware MPI boundary exchange */
int GPUMPIExchangeBoundariesnD(
  int ndims,
  int nvars,
  int *dim,
  int ghosts,
  void *m,
  double *var
);

/* Internal GPU kernels for packing/unpacking MPI halo regions
   These are always declared but only implemented for GPU builds.
   In non-GPU builds, they are no-op stubs that should never be called. */
void gpu_launch_mpi_pack_boundary(
  const double *var,
  double *buf,
  int ndims,
  int nvars,
  const int *dim,
  int ghosts,
  int dir,
  int side,
  int blockSize
);

void gpu_launch_mpi_unpack_boundary(
  double *var,
  const double *buf,
  int ndims,
  int nvars,
  const int *dim,
  int ghosts,
  int dir,
  int side,
  int blockSize
);

/* Stream-aware versions for overlapping pack/unpack with transfers */
void gpu_launch_mpi_pack_boundary_stream(
  const double *var,
  double *buf,
  int ndims,
  int nvars,
  const int *dim,
  int ghosts,
  int dir,
  int side,
  int blockSize,
  void *stream
);

void gpu_launch_mpi_unpack_boundary_stream(
  double *var,
  const double *buf,
  int ndims,
  int nvars,
  const int *dim,
  int ghosts,
  int dir,
  int side,
  int blockSize,
  void *stream
);

/* MPI stream management (one stream per dimension face) */
int GPUCreateMPIStreams(void **streams, int nstreams);
int GPUDestroyMPIStreams(void **streams, int nstreams);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_MPI_H_ */

