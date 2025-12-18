/*! @file gpu_mpi_kernels.cu
    @brief GPU kernels for packing/unpacking MPI halo regions (ndims <= 3)
*/

#include <gpu.h>
#include <gpu_launch.h>

/* Note: dim[] is a host pointer (small). We pass it as a device kernel argument;
   CUDA/HIP will copy the small array by value when launching (pointer value still host),
   so DO NOT dereference dim[] on device. Instead, copy dim entries to scalars on host
   and pass them individually. To keep interface simple, we copy dim into local scalars
   in the launch wrapper and pass scalars to kernels. */

static __device__ __forceinline__ int idx1dwo_1d(int i0, int dim0, int ghosts, int off0)
{
  return (i0 + ghosts + off0);
}

static __device__ __forceinline__ int idx1dwo_2d(int i0, int i1, int dim0, int dim1, int ghosts, int off0, int off1)
{
  int index = (i1 + ghosts + off1);
  index = index * (dim0 + 2*ghosts) + (i0 + ghosts + off0);
  return index;
}

static __device__ __forceinline__ int idx1dwo_3d(int i0, int i1, int i2,
                                                 int dim0, int dim1, int dim2,
                                                 int ghosts,
                                                 int off0, int off1, int off2)
{
  int index = (i2 + ghosts + off2);
  index = index * (dim1 + 2*ghosts) + (i1 + ghosts + off1);
  index = index * (dim0 + 2*ghosts) + (i0 + ghosts + off0);
  return index;
}

__global__ void gpu_mpi_pack_boundary_kernel(
  const double *var,
  double *buf,
  int ndims,
  int nvars,
  int dim0, int dim1, int dim2,
  int ghosts,
  int dir,
  int side /* -1 left, +1 right */,
  int npts /* number of points in the packed region (excluding nvars) */
)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= npts) return;

  /* bounds: bounds[dir] = ghosts; bounds[other] = dim[other] */
  int b0 = (dir == 0 ? ghosts : dim0);
  int b1 = (ndims > 1 ? (dir == 1 ? ghosts : dim1) : 1);
  /* b2 not needed explicitly; tid decoding uses b0/b1 and remaining quotient. */

  /* decode tid into i0,i1,i2 within bounds with index0 fastest */
  int i0 = 0, i1 = 0, i2 = 0;
  if (ndims == 1) {
    i0 = tid;
  } else if (ndims == 2) {
    i0 = tid % b0;
    i1 = tid / b0;
  } else { /* ndims == 3 */
    int t = tid;
    i0 = t % b0; t /= b0;
    i1 = t % b1; t /= b1;
    i2 = t;
  }

  /* offset: 0 for all dims except dir.
     For packing: left -> off_dir=0, right -> off_dir=dim[dir]-ghosts */
  int off0 = 0, off1 = 0, off2 = 0;
  int off_dir = 0;
  if (side > 0) {
    off_dir = (dir == 0 ? (dim0 - ghosts) : (dir == 1 ? (dim1 - ghosts) : (dim2 - ghosts)));
  }
  if (dir == 0) off0 = off_dir;
  else if (dir == 1) off1 = off_dir;
  else off2 = off_dir;

  int p1 = 0;
  if (ndims == 1) p1 = idx1dwo_1d(i0, dim0, ghosts, off0);
  else if (ndims == 2) p1 = idx1dwo_2d(i0, i1, dim0, dim1, ghosts, off0, off1);
  else p1 = idx1dwo_3d(i0, i1, i2, dim0, dim1, dim2, ghosts, off0, off1, off2);

  /* p2 is tid in bounds ordering; buf stores nvars contiguous for each point */
  int p2 = tid;
  const int base_var = nvars * p1;
  const int base_buf = nvars * p2;
  for (int v = 0; v < nvars; v++) {
    buf[base_buf + v] = var[base_var + v];
  }
}

__global__ void gpu_mpi_unpack_boundary_kernel(
  double *var,
  const double *buf,
  int ndims,
  int nvars,
  int dim0, int dim1, int dim2,
  int ghosts,
  int dir,
  int side /* -1 left, +1 right */,
  int npts
)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= npts) return;

  int b0 = (dir == 0 ? ghosts : dim0);
  int b1 = (ndims > 1 ? (dir == 1 ? ghosts : dim1) : 1);
  /* b2 not needed explicitly; tid decoding uses b0/b1 and remaining quotient. */

  int i0 = 0, i1 = 0, i2 = 0;
  if (ndims == 1) {
    i0 = tid;
  } else if (ndims == 2) {
    i0 = tid % b0;
    i1 = tid / b0;
  } else {
    int t = tid;
    i0 = t % b0; t /= b0;
    i1 = t % b1; t /= b1;
    i2 = t;
  }

  /* offset: 0 for all dims except dir.
     For unpacking into ghost points: left -> off_dir=-ghosts, right -> off_dir=dim[dir] */
  int off0 = 0, off1 = 0, off2 = 0;
  int off_dir = 0;
  if (side < 0) off_dir = -ghosts;
  else off_dir = (dir == 0 ? dim0 : (dir == 1 ? dim1 : dim2));

  if (dir == 0) off0 = off_dir;
  else if (dir == 1) off1 = off_dir;
  else off2 = off_dir;

  int p1 = 0;
  if (ndims == 1) p1 = idx1dwo_1d(i0, dim0, ghosts, off0);
  else if (ndims == 2) p1 = idx1dwo_2d(i0, i1, dim0, dim1, ghosts, off0, off1);
  else p1 = idx1dwo_3d(i0, i1, i2, dim0, dim1, dim2, ghosts, off0, off1, off2);

  int p2 = tid;
  const int base_var = nvars * p1;
  const int base_buf = nvars * p2;
  for (int v = 0; v < nvars; v++) {
    var[base_var + v] = buf[base_buf + v];
  }
}

extern "C" void gpu_launch_mpi_pack_boundary(
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
  if (!var || !buf || !dim) return;
  if (ndims < 1 || ndims > 3) return;
  if (dir < 0 || dir >= ndims) return;

  const int dim0 = dim[0];
  const int dim1 = (ndims > 1) ? dim[1] : 1;
  const int dim2 = (ndims > 2) ? dim[2] : 1;

  int npts = 1;
  for (int d = 0; d < ndims; d++) {
    if (d == dir) npts *= ghosts;
    else npts *= dim[d];
  }

  GPULaunchConfig cfg = GPUConfigureLaunch((size_t)npts, blockSize);
  gpu_mpi_pack_boundary_kernel<<<cfg.gridSize, cfg.blockSize>>>(
    var, buf, ndims, nvars, dim0, dim1, dim2, ghosts, dir, side, npts
  );
}

extern "C" void gpu_launch_mpi_unpack_boundary(
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
  if (!var || !buf || !dim) return;
  if (ndims < 1 || ndims > 3) return;
  if (dir < 0 || dir >= ndims) return;

  const int dim0 = dim[0];
  const int dim1 = (ndims > 1) ? dim[1] : 1;
  const int dim2 = (ndims > 2) ? dim[2] : 1;

  int npts = 1;
  for (int d = 0; d < ndims; d++) {
    if (d == dir) npts *= ghosts;
    else npts *= dim[d];
  }

  GPULaunchConfig cfg = GPUConfigureLaunch((size_t)npts, blockSize);
  gpu_mpi_unpack_boundary_kernel<<<cfg.gridSize, cfg.blockSize>>>(
    var, buf, ndims, nvars, dim0, dim1, dim2, ghosts, dir, side, npts
  );
}


