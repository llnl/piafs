/*! @file gpu_mpi.c
    @brief GPU-aware MPI communication functions
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_mpi.h>
#include <mpivars.h>
#include <arrayfunctions.h>
#include <basic.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef serial
#include <mpi.h>
/* Check for OpenMPI CUDA-aware extensions */
#ifdef OPEN_MPI
#include <mpi-ext.h>
#endif
#endif

/* Static GPU buffers for GPU-aware MPI */
#ifdef GPU_AWARE_MPI
static double *d_sendbuf = NULL;
static double *d_recvbuf = NULL;
static size_t d_mpi_buf_capacity = 0;

/* Runtime check for GPU-aware MPI support
   -1 = not checked yet, 0 = not supported, 1 = supported */
static int gpu_aware_mpi_available = -1;

/* Check if MPI library actually supports GPU-aware operations at runtime */
static int check_gpu_aware_mpi_support(void)
{
  if (gpu_aware_mpi_available >= 0) {
    return gpu_aware_mpi_available;
  }

  /* Check environment variable override first */
  const char *env = getenv("PIAFS_FORCE_GPU_AWARE_MPI");
  if (env) {
    if (env[0] == '1' || env[0] == 'y' || env[0] == 'Y') {
      gpu_aware_mpi_available = 1;
      return 1;
    } else if (env[0] == '0' || env[0] == 'n' || env[0] == 'N') {
      gpu_aware_mpi_available = 0;
      return 0;
    }
  }

  /* Try to detect GPU-aware MPI support from the MPI library */
  int supported = 0;

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  /* OpenMPI compile-time check passed, now check runtime */
  supported = MPIX_Query_cuda_support();
#elif defined(MVAPICH2_VERSION)
  /* MVAPICH2: check environment variable */
  const char *mv2_cuda = getenv("MV2_USE_CUDA");
  if (mv2_cuda && mv2_cuda[0] == '1') {
    supported = 1;
  }
#else
  /* Unknown MPI implementation - default to disabled for safety */
  supported = 0;
#endif

  gpu_aware_mpi_available = supported;

  /* Print diagnostic message (only rank 0) */
  int rank = 0;
#ifndef serial
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank == 0) {
    if (supported) {
      printf("GPU-aware MPI: enabled (runtime check passed)\n");
    } else {
      printf("GPU-aware MPI: disabled (runtime check failed, using host staging)\n");
      printf("  Set PIAFS_FORCE_GPU_AWARE_MPI=1 to override if your MPI is CUDA-aware.\n");
    }
  }

  return supported;
}

static int ensure_gpu_mpi_buffers(size_t required_size)
{
  if (required_size <= d_mpi_buf_capacity) return 0;

  /* Free old buffers */
  if (d_sendbuf) { GPUFree(d_sendbuf); d_sendbuf = NULL; }
  if (d_recvbuf) { GPUFree(d_recvbuf); d_recvbuf = NULL; }

  /* Allocate new buffers */
  if (GPUAllocate((void**)&d_sendbuf, required_size)) {
    fprintf(stderr, "Error: Failed to allocate GPU send buffer for GPU-aware MPI (%zu bytes)\n", required_size);
    d_mpi_buf_capacity = 0;
    return 1;
  }
  if (GPUAllocate((void**)&d_recvbuf, required_size)) {
    fprintf(stderr, "Error: Failed to allocate GPU recv buffer for GPU-aware MPI (%zu bytes)\n", required_size);
    GPUFree(d_sendbuf);
    d_sendbuf = NULL;
    d_mpi_buf_capacity = 0;
    return 1;
  }

  d_mpi_buf_capacity = required_size;
  return 0;
}
#endif

/* GPU-aware MPI boundary exchange
   For GPU arrays, this function:
   - With GPU_AWARE_MPI: Uses device buffers directly with MPI (no host staging)
   - Without GPU_AWARE_MPI: Copies ghost points from GPU to host send buffers,
     performs MPI communication on host, copies received data from host to GPU
*/
int GPUMPIExchangeBoundariesnD(
  int ndims,
  int nvars,
  int *dim,
  int ghosts,
  void *m,
  double *var  /* GPU array */
)
{
#ifndef serial
  MPIVariables *mpi = (MPIVariables*) m;
  int d;

  int *ip = mpi->ip;
  int *iproc = mpi->iproc;
  int *bcflag = mpi->bcperiodic;

  int neighbor_rank[2*ndims], nip[ndims], index[ndims], bounds[ndims], offset[ndims];
  MPI_Request rcvreq[2*ndims], sndreq[2*ndims];
  for (d = 0; d < 2*ndims; d++) rcvreq[d] = sndreq[d] = MPI_REQUEST_NULL;

  /* Calculate neighbor ranks */
  for (d = 0; d < ndims; d++) {
    _ArrayCopy1D_(ip, nip, ndims);
    if (ip[d] == 0) nip[d] = iproc[d] - 1;
    else nip[d]--;
    if ((ip[d] == 0) && (!bcflag[d])) neighbor_rank[2*d] = -1;
    else neighbor_rank[2*d] = MPIRank1D(ndims, iproc, nip);

    _ArrayCopy1D_(ip, nip, ndims);
    if (ip[d] == (iproc[d] - 1)) nip[d] = 0;
    else nip[d]++;
    if ((ip[d] == (iproc[d] - 1)) && (!bcflag[d])) neighbor_rank[2*d+1] = -1;
    else neighbor_rank[2*d+1] = MPIRank1D(ndims, iproc, nip);
  }

  /* Calculate buffer dimensions */
  int stride = mpi->maxbuf;
  int bufdim[ndims];
  for (d = 0; d < ndims; d++) {
    bufdim[d] = 1;
    int i;
    for (i = 0; i < ndims; i++) {
      if (i == d) bufdim[d] *= ghosts;
      else bufdim[d] *= dim[i];
    }
  }

#ifdef GPU_AWARE_MPI
  /* GPU-aware MPI path: use device buffers directly */
  if (GPUShouldUse() && check_gpu_aware_mpi_support()) {
    /* Ensure GPU buffers are large enough */
    size_t total_buf_size = (size_t)(2 * ndims) * (size_t)stride * sizeof(double);
    if (ensure_gpu_mpi_buffers(total_buf_size)) {
      return 1;
    }

    /* Post receive requests using GPU buffers */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        MPI_Irecv(&d_recvbuf[2*d*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d], 1630,
                  mpi->world, &rcvreq[2*d]);
      }
      if (neighbor_rank[2*d+1] != -1) {
        MPI_Irecv(&d_recvbuf[(2*d+1)*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d+1], 1631,
                  mpi->world, &rcvreq[2*d+1]);
      }
    }

    /* Pack data directly to GPU send buffers */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        gpu_launch_mpi_pack_boundary(var, &d_sendbuf[2*d*stride], ndims, nvars, dim, ghosts, d, -1, 256);
      }
      if (neighbor_rank[2*d+1] != -1) {
        gpu_launch_mpi_pack_boundary(var, &d_sendbuf[(2*d+1)*stride], ndims, nvars, dim, ghosts, d, +1, 256);
      }
    }

    /* Synchronize to ensure packing is complete before MPI sends */
    GPUSync();

    /* Send data using GPU buffers */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        MPI_Isend(&d_sendbuf[2*d*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d], 1631,
                  mpi->world, &sndreq[2*d]);
      }
      if (neighbor_rank[2*d+1] != -1) {
        MPI_Isend(&d_sendbuf[(2*d+1)*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d+1], 1630,
                  mpi->world, &sndreq[2*d+1]);
      }
    }

    /* Wait for receives */
    MPI_Status status_arr[2*ndims];
    MPI_Waitall(2*ndims, rcvreq, status_arr);

    /* Unpack received data directly from GPU buffers */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        gpu_launch_mpi_unpack_boundary(var, &d_recvbuf[2*d*stride], ndims, nvars, dim, ghosts, d, -1, 256);
      }
      if (neighbor_rank[2*d+1] != -1) {
        gpu_launch_mpi_unpack_boundary(var, &d_recvbuf[(2*d+1)*stride], ndims, nvars, dim, ghosts, d, +1, 256);
      }
    }

    /* Wait for sends to complete */
    MPI_Waitall(2*ndims, sndreq, status_arr);

    return 0;
  }
#endif

  /* Non-GPU-aware MPI path (or CPU path) */
  double *sendbuf = mpi->sendbuf;
  double *recvbuf = mpi->recvbuf;

  /* Post receive requests */
  for (d = 0; d < ndims; d++) {
    if (neighbor_rank[2*d] != -1) {
      MPI_Irecv(&recvbuf[2*d*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d], 1630,
                mpi->world, &rcvreq[2*d]);
    }
    if (neighbor_rank[2*d+1] != -1) {
      MPI_Irecv(&recvbuf[(2*d+1)*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d+1], 1631,
                mpi->world, &rcvreq[2*d+1]);
    }
  }

  /* Copy data from GPU to host send buffers */
  if (GPUShouldUse()) {
    /* Static device buffer for packing - sized to hold all faces */
    static double *d_packbuf = NULL;
    static size_t d_packbuf_capacity = 0;
    size_t total_buf_size = (size_t)(2 * ndims) * (size_t)stride;
    if (total_buf_size > d_packbuf_capacity) {
      if (d_packbuf) GPUFree(d_packbuf);
      if (GPUAllocate((void**)&d_packbuf, total_buf_size * sizeof(double))) {
        fprintf(stderr, "Error: GPUMPIExchangeBoundariesnD: failed to allocate device pack buffer (%zu bytes)\n",
                total_buf_size * sizeof(double));
        d_packbuf = NULL;
        d_packbuf_capacity = 0;
        return 1;
      }
      d_packbuf_capacity = total_buf_size;
    }

    /* Pack ALL boundaries to device buffer first (overlaps kernel launches) */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        gpu_launch_mpi_pack_boundary(var, &d_packbuf[2*d*stride], ndims, nvars, dim, ghosts, d, -1, 256);
      }
      if (neighbor_rank[2*d+1] != -1) {
        gpu_launch_mpi_pack_boundary(var, &d_packbuf[(2*d+1)*stride], ndims, nvars, dim, ghosts, d, +1, 256);
      }
    }

    /* Sync to ensure all packing is complete */
    GPUSync();

    /* Now copy all packed data to host (with pinned memory this is fast) */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        const int bufsize = bufdim[d] * nvars;
        GPUCopyToHost(&sendbuf[2*d*stride], &d_packbuf[2*d*stride], (size_t)bufsize * sizeof(double));
      }
      if (neighbor_rank[2*d+1] != -1) {
        const int bufsize = bufdim[d] * nvars;
        GPUCopyToHost(&sendbuf[(2*d+1)*stride], &d_packbuf[(2*d+1)*stride], (size_t)bufsize * sizeof(double));
      }
    }
  } else {
    /* CPU path - use original logic */
    for (d = 0; d < ndims; d++) {
      _ArrayCopy1D_(dim, bounds, ndims);
      bounds[d] = ghosts;
      if (neighbor_rank[2*d] != -1) {
        _ArraySetValue_(offset, ndims, 0);
        int done = 0;
        _ArraySetValue_(index, ndims, 0);
        while (!done) {
          int p1;
          _ArrayIndex1DWO_(ndims, dim, index, offset, ghosts, p1);
          int p2;
          _ArrayIndex1D_(ndims, bounds, index, 0, p2);
          _ArrayCopy1D_((var + nvars*p1), (sendbuf + 2*d*stride + nvars*p2), nvars);
          _ArrayIncrementIndex_(ndims, bounds, index, done);
        }
      }
      if (neighbor_rank[2*d+1] != -1) {
        _ArraySetValue_(offset, ndims, 0);
        offset[d] = dim[d] - ghosts;
        int done = 0;
        _ArraySetValue_(index, ndims, 0);
        while (!done) {
          int p1;
          _ArrayIndex1DWO_(ndims, dim, index, offset, ghosts, p1);
          int p2;
          _ArrayIndex1D_(ndims, bounds, index, 0, p2);
          _ArrayCopy1D_((var + nvars*p1), (sendbuf + (2*d+1)*stride + nvars*p2), nvars);
          _ArrayIncrementIndex_(ndims, bounds, index, done);
        }
      }
    }
  }

  /* Send data */
  for (d = 0; d < ndims; d++) {
    if (neighbor_rank[2*d] != -1) {
      MPI_Isend(&sendbuf[2*d*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d], 1631,
                mpi->world, &sndreq[2*d]);
    }
    if (neighbor_rank[2*d+1] != -1) {
      MPI_Isend(&sendbuf[(2*d+1)*stride], bufdim[d]*nvars, MPI_DOUBLE, neighbor_rank[2*d+1], 1630,
                mpi->world, &sndreq[2*d+1]);
    }
  }

  /* Wait for receives */
  MPI_Status status_arr[2*ndims];
  MPI_Waitall(2*ndims, rcvreq, status_arr);

  /* Copy received data from host to GPU ghost points */
  if (GPUShouldUse()) {
    /* Static device buffer for unpacking - sized to hold all faces */
    static double *d_unpackbuf = NULL;
    static size_t d_unpackbuf_capacity = 0;
    size_t total_buf_size = (size_t)(2 * ndims) * (size_t)stride;
    if (total_buf_size > d_unpackbuf_capacity) {
      if (d_unpackbuf) GPUFree(d_unpackbuf);
      if (GPUAllocate((void**)&d_unpackbuf, total_buf_size * sizeof(double))) {
        fprintf(stderr, "Error: GPUMPIExchangeBoundariesnD: failed to allocate device unpack buffer (%zu bytes)\n",
                total_buf_size * sizeof(double));
        d_unpackbuf = NULL;
        d_unpackbuf_capacity = 0;
        return 1;
      }
      d_unpackbuf_capacity = total_buf_size;
    }

    /* Copy all received data to device (with pinned memory this is fast) */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        const int bufsize = bufdim[d] * nvars;
        GPUCopyToDevice(&d_unpackbuf[2*d*stride], &recvbuf[2*d*stride], (size_t)bufsize * sizeof(double));
      }
      if (neighbor_rank[2*d+1] != -1) {
        const int bufsize = bufdim[d] * nvars;
        GPUCopyToDevice(&d_unpackbuf[(2*d+1)*stride], &recvbuf[(2*d+1)*stride], (size_t)bufsize * sizeof(double));
      }
    }

    /* Launch all unpack kernels (they can overlap on the GPU) */
    for (d = 0; d < ndims; d++) {
      if (neighbor_rank[2*d] != -1) {
        gpu_launch_mpi_unpack_boundary(var, &d_unpackbuf[2*d*stride], ndims, nvars, dim, ghosts, d, -1, 256);
      }
      if (neighbor_rank[2*d+1] != -1) {
        gpu_launch_mpi_unpack_boundary(var, &d_unpackbuf[(2*d+1)*stride], ndims, nvars, dim, ghosts, d, +1, 256);
      }
    }
  } else {
    /* CPU path */
    for (d = 0; d < ndims; d++) {
      _ArrayCopy1D_(dim, bounds, ndims);
      bounds[d] = ghosts;
      if (neighbor_rank[2*d] != -1) {
        _ArraySetValue_(offset, ndims, 0);
        offset[d] = -ghosts;
        int done = 0;
        _ArraySetValue_(index, ndims, 0);
        while (!done) {
          int p1;
          _ArrayIndex1DWO_(ndims, dim, index, offset, ghosts, p1);
          int p2;
          _ArrayIndex1D_(ndims, bounds, index, 0, p2);
          _ArrayCopy1D_((recvbuf + 2*d*stride + nvars*p2), (var + nvars*p1), nvars);
          _ArrayIncrementIndex_(ndims, bounds, index, done);
        }
      }
      if (neighbor_rank[2*d+1] != -1) {
        _ArraySetValue_(offset, ndims, 0);
        offset[d] = dim[d];
        int done = 0;
        _ArraySetValue_(index, ndims, 0);
        while (!done) {
          int p1;
          _ArrayIndex1DWO_(ndims, dim, index, offset, ghosts, p1);
          int p2;
          _ArrayIndex1D_(ndims, bounds, index, 0, p2);
          _ArrayCopy1D_((recvbuf + (2*d+1)*stride + nvars*p2), (var + nvars*p1), nvars);
          _ArrayIncrementIndex_(ndims, bounds, index, done);
        }
      }
    }
  }

  /* Wait for sends */
  MPI_Waitall(2*ndims, sndreq, status_arr);

#endif
  return 0;
}
