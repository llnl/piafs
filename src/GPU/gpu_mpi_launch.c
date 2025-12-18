/*! @file gpu_mpi_launch.c
    @brief CPU fallback for GPU MPI kernel launch wrappers
    @details These are no-op stubs used when building without GPU support.
             They should never actually be called (GPUShouldUse() guards the calls).
*/

#include <gpu_mpi.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(GPU_CUDA) && !defined(GPU_HIP)

/* CPU fallback: These should never be called in CPU-only builds */
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
)
{
  fprintf(stderr, "ERROR: gpu_launch_mpi_pack_boundary called but GPU support not compiled!\n");
  exit(1);
}

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
)
{
  fprintf(stderr, "ERROR: gpu_launch_mpi_unpack_boundary called but GPU support not compiled!\n");
  exit(1);
}

#endif /* !defined(GPU_CUDA) && !defined(GPU_HIP) */

