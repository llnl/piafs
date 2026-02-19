/*! @file gpu_kernels.cu
    @brief GPU kernels for array operations (CUDA/HIP compatible)
*/

#include <gpu.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Kernel: Array copy */
GPU_KERNEL void gpu_array_copy(double *dst, const double *src, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

/* Kernel: Array set value */
GPU_KERNEL void gpu_array_set_value(double *x, double value, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = value;
  }
}

/* Kernel: Array scale */
GPU_KERNEL void gpu_array_scale(double *x, double a, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] *= a;
  }
}

/* Kernel: Array AXPY: y = a*x + y */
GPU_KERNEL void gpu_array_axpy(const double *x, double a, double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] += a * x[idx];
  }
}

/* Kernel: Array AYPX: y = a*y + x */
GPU_KERNEL void gpu_array_aypx(const double *x, double a, double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * y[idx] + x[idx];
  }
}

/* Kernel: Array AXBY: z = a*x + b*y */
GPU_KERNEL void gpu_array_axby(double *z, double a, const double *x, double b, const double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = a * x[idx] + b * y[idx];
  }
}

/* Kernel: Array scale and copy: y = a*x */
GPU_KERNEL void gpu_array_scale_copy(const double *x, double a, double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx];
  }
}

/* Kernel: Array add: z = x + y */
GPU_KERNEL void gpu_array_add(double *z, const double *x, const double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = x[idx] + y[idx];
  }
}

/* Kernel: Array subtract: z = x - y */
GPU_KERNEL void gpu_array_subtract(double *z, const double *x, const double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = x[idx] - y[idx];
  }
}

/* Kernel: Array multiply: z = x * y */
GPU_KERNEL void gpu_array_multiply(double *z, const double *x, const double *y, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = x[idx] * y[idx];
  }
}

/* Kernel: Block-level max reduction
   Each block computes the max of its portion and writes to partial_max[blockIdx.x]
   Uses shared memory for efficient reduction within a block.
*/
GPU_KERNEL void gpu_array_max_block(const double *x, double *partial_max, int n)
{
  extern __shared__ double sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Load element or -infinity if out of bounds */
  sdata[tid] = (idx < n) ? x[idx] : -1e308;
  __syncthreads();

  /* Parallel reduction in shared memory */
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }

  /* Write result for this block */
  if (tid == 0) {
    partial_max[blockIdx.x] = sdata[0];
  }
}

/* Kernel: Final reduction of partial max values
   Reduces an array of partial max values to a single max.
   n is the number of partial results (number of blocks from first pass).
*/
GPU_KERNEL void gpu_array_max_final(const double *partial_max, double *result, int n)
{
  extern __shared__ double sdata[];

  int tid = threadIdx.x;

  /* Load elements, handling case where n > blockDim.x */
  double local_max = -1e308;
  for (int i = tid; i < n; i += blockDim.x) {
    if (partial_max[i] > local_max) {
      local_max = partial_max[i];
    }
  }
  sdata[tid] = local_max;
  __syncthreads();

  /* Parallel reduction in shared memory */
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }

  /* Write final result */
  if (tid == 0) {
    result[0] = sdata[0];
  }
}

/* Kernel: Hyperbolic flux derivative computation */
GPU_KERNEL void gpu_hyperbolic_flux_derivative(
  double *hyp,           /* output: hyperbolic term */
  const double *fluxI,   /* input: interface fluxes */
  const double *dxinv,   /* input: 1/dx */
  int nvars,             /* number of variables */
  int ndims,             /* number of dimensions */
  int *dim,              /* dimensions */
  int ghosts,            /* ghost points */
  int dir,               /* current direction */
  int offset             /* offset in dxinv array */
)
{
  /* This is a simplified version - full implementation would need
     proper multi-dimensional indexing */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int npoints = 1;
  for (int d = 0; d < ndims; d++) npoints *= dim[d];

  if (idx < npoints) {
    /* Compute multi-dimensional index from linear index */
    int index[3] = {0, 0, 0};
    int temp = idx;
    for (int d = ndims-1; d >= 0; d--) {
      index[d] = temp % dim[d];
      temp /= dim[d];
    }

    /* Compute 1D index with ghosts */
    int p = index[ndims-1] + ghosts;
    for (int d = ndims-2; d >= 0; d--) {
      p = p * (dim[d] + 2*ghosts) + (index[d] + ghosts);
    }

    /* Compute interface indices */
    int index1[3], index2[3];
    for (int d = 0; d < ndims; d++) {
      index1[d] = index[d];
      index2[d] = index[d];
    }
    index2[dir]++;

    /* Compute interface 1D indices (no ghosts) */
    int p1 = index1[ndims-1];
    int p2 = index2[ndims-1];
    for (int d = ndims-2; d >= 0; d--) {
      int dim_interface = (d == dir) ? dim[d] + 1 : dim[d];
      p1 = p1 * dim_interface + index1[d];
      p2 = p2 * dim_interface + index2[d];
    }

    /* Compute derivative */
    double dx = dxinv[offset + ghosts + index[dir]];
    for (int v = 0; v < nvars; v++) {
      hyp[p*nvars + v] += dx * (fluxI[p2*nvars + v] - fluxI[p1*nvars + v]);
    }
  }
}

