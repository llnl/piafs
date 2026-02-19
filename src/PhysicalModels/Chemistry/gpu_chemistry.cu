/*! @file gpu_chemistry.cu
    @brief GPU kernels for chemistry computations
    @author Debojyoti Ghosh, Albertine Oudin
*/

#include <gpu.h>
#include <physicalmodels/chemistry.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
  #define GPU_DEVICE __device__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
  #define GPU_DEVICE __device__
#else
  #define GPU_KERNEL
  #define GPU_DEVICE
#endif

/*! Device function to compute photon density at first z-layer */
GPU_DEVICE double gpu_hnu_first_layer(
  double I0,
  double c,
  double h,
  double nu,
  double n_O2,
  double t_pulse_norm,
  double t_start_norm,
  double imap_p,
  double t
)
{
  double sigma = t_pulse_norm / 2.35482;
  double I0_val = 0.0;
  if (t > t_start_norm) {
    double tp = t - (t_start_norm + t_pulse_norm);
    I0_val = I0 * exp(-(tp * tp) / (2 * sigma * sigma)) * imap_p;
  }
  return I0_val / (c * h * nu * n_O2);
}

/*! Device function to compute damping factor for photon density */
GPU_DEVICE double gpu_hnu_damp_factor(
  double sO3,
  double n_O2,
  double dz,
  double nO3
)
{
  double sigma = sO3 * n_O2;
  double damp_fac = 1.0 - dz * sigma * nO3;
  return damp_fac;
}

/*! GPU kernel to set photon density for 1D/2D cases
 *  Each thread handles one grid point and sequentially computes z-layers
 */
GPU_KERNEL void gpu_chemistry_photon_density_1d2d_kernel(
  double* __restrict__ nv_hnu,
  const double* __restrict__ u,
  const double* __restrict__ imap,
  int npoints,
  int grid_stride,
  int z_stride,
  int n_flow_vars,
  int nz,
  double I0,
  double c,
  double h,
  double nu,
  double n_O2_chem,
  double t_pulse_norm,
  double t_start_norm,
  double sO3,
  double dz,
  double t
)
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npoints) return;

  // First z-layer (iz = 0) from prescribed function
  double imap_p = imap[p];
  nv_hnu[nz * p + 0] = gpu_hnu_first_layer(
    I0, c, h, nu, n_O2_chem, t_pulse_norm, t_start_norm, imap_p, t
  );

  // Remaining z-layers computed sequentially with damping
  for (int iz = 1; iz < nz; iz++) {
    double n_O3 = u[grid_stride * p + n_flow_vars + z_stride * (iz - 1) + iO3];
    nv_hnu[nz * p + iz] = nv_hnu[nz * p + (iz - 1)] * gpu_hnu_damp_factor(sO3, n_O2_chem, dz, n_O3);
  }
}

/*! GPU kernel to set photon density for first k-layer in 3D (k=0)
 *  Parallelized over i,j grid points
 */
GPU_KERNEL void gpu_chemistry_photon_density_3d_first_layer_kernel(
  double* __restrict__ nv_hnu,
  const double* __restrict__ imap,
  int imax,
  int jmax,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  double I0,
  double c,
  double h,
  double nu,
  double n_O2_chem,
  double t_pulse_norm,
  double t_start_norm,
  double t
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imax || j >= jmax) return;

  int k = 0;
  // Compute 1D index
  int p = (i + ghosts) + (dim0 + 2 * ghosts) * ((j + ghosts) + (dim1 + 2 * ghosts) * (k + ghosts));

  double imap_p = imap[p];
  nv_hnu[p] = gpu_hnu_first_layer(
    I0, c, h, nu, n_O2_chem, t_pulse_norm, t_start_norm, imap_p, t
  );
}

/*! GPU kernel to set photon density for subsequent k-layers in 3D (k > 0)
 *  Parallelized over i,j grid points, sequentially over k
 *  NOTE: This is the legacy per-layer kernel, kept for compatibility.
 *  Use gpu_chemistry_photon_density_3d_batched_kernel for better performance.
 */
GPU_KERNEL void gpu_chemistry_photon_density_3d_next_layer_kernel(
  double* __restrict__ nv_hnu,
  const double* __restrict__ u,
  int imax,
  int jmax,
  int k,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  int grid_stride,
  int n_flow_vars,
  double sO3,
  double n_O2_chem,
  double dz
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imax || j >= jmax) return;

  // Compute 1D indices for current layer (k) and previous layer (k-1)
  int p_km1 = (i + ghosts) + (dim0 + 2 * ghosts) * ((j + ghosts) + (dim1 + 2 * ghosts) * (k - 1 + ghosts));
  int p     = (i + ghosts) + (dim0 + 2 * ghosts) * ((j + ghosts) + (dim1 + 2 * ghosts) * (k + ghosts));

  // Get O3 concentration from previous layer
  double n_O3 = u[grid_stride * p_km1 + n_flow_vars + iO3];

  // Compute damped photon density
  nv_hnu[p] = nv_hnu[p_km1] * gpu_hnu_damp_factor(sO3, n_O2_chem, dz, n_O3);
}

/*! GPU kernel to compute ALL z-layers of photon density in 3D in a SINGLE launch
 *  Each thread handles one (i,j) column and iterates through all z-layers.
 *  This eliminates ~(kmax-1) kernel launches per time step.
 */
GPU_KERNEL void gpu_chemistry_photon_density_3d_batched_kernel(
  double* __restrict__ nv_hnu,
  const double* __restrict__ u,
  const double* __restrict__ imap,
  int imax,
  int jmax,
  int kmax,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  int grid_stride,
  int n_flow_vars,
  double I0,
  double c,
  double h,
  double nu,
  double n_O2_chem,
  double t_pulse_norm,
  double t_start_norm,
  double sO3,
  double dz,
  double t,
  int first_rank_z,  /* 1 if this is the first rank in z-direction */
  int kstart         /* Starting k index (0 if first_rank_z, else 0 but expects ghost data) */
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imax || j >= jmax) return;

  const int stride_j = dim0 + 2 * ghosts;
  const int stride_k = stride_j * (dim1 + 2 * ghosts);
  const int ij_base = (i + ghosts) + stride_j * (j + ghosts);

  // First layer (k=0) - only compute if this is the first rank in z
  if (first_rank_z) {
    int p0 = ij_base + stride_k * ghosts;
    double imap_p = imap[p0];
    nv_hnu[p0] = gpu_hnu_first_layer(I0, c, h, nu, n_O2_chem, t_pulse_norm, t_start_norm, imap_p, t);
  }

  // Subsequent layers - each depends on the previous layer
  // Use register to cache previous layer's photon density
  int p_prev = ij_base + stride_k * (kstart - 1 + ghosts);
  double nv_hnu_prev = nv_hnu[p_prev];

  #pragma unroll 4
  for (int k = kstart; k < kmax; k++) {
    int p_km1 = ij_base + stride_k * (k - 1 + ghosts);
    int p     = ij_base + stride_k * (k + ghosts);

    // Get O3 concentration from previous layer
    double n_O3 = u[grid_stride * p_km1 + n_flow_vars + iO3];

    // Compute damped photon density using cached value
    double nv_hnu_curr = nv_hnu_prev * gpu_hnu_damp_factor(sO3, n_O2_chem, dz, n_O3);
    nv_hnu[p] = nv_hnu_curr;

    // Update cache for next iteration
    nv_hnu_prev = nv_hnu_curr;
  }
}

/*! GPU kernel to compute chemistry source terms for reaction species */
GPU_KERNEL void gpu_chemistry_source_kernel(
  double* __restrict__ source,
  const double* __restrict__ u,
  const double* __restrict__ nv_hnu,
  int npoints,
  int nvars,
  int n_flow_vars,
  int grid_stride,
  int z_stride,
  int z_i,
  int ndims,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  double k0a, double k0b, double k1a, double k1b,
  double k2a, double k2b, double k3a, double k3b,
  double k4, double k5, double k6,
  double q0a, double q0b, double q1a, double q1b,
  double q2a, double q2b, double q3a, double q3b,
  double q4, double q5, double q6,
  double gamma_m1_inv
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npoints) return;

  /* Check if this is an interior point (not a ghost) */
  if (ndims == 2) {
    /* 2D case: compute i,j from idx */
    int stride_j = dim0 + 2*ghosts;
    int j = idx / stride_j;
    int i = idx - j * stride_j;
    if (i < ghosts || i >= dim0+ghosts || j < ghosts || j >= dim1+ghosts) {
      return; /* Ghost point, skip */
    }
  } else if (ndims == 3) {
    /* 3D case: compute i,j,k from idx */
    int stride_j = dim0 + 2*ghosts;
    int stride_k = (dim0 + 2*ghosts) * (dim1 + 2*ghosts);
    int k = idx / stride_k;
    int rem = idx - k * stride_k;
    int j = rem / stride_j;
    int i = rem - j * stride_j;
    if (i < ghosts || i >= dim0+ghosts ||
        j < ghosts || j >= dim1+ghosts ||
        k < ghosts || k >= dim2+ghosts) {
      return; /* Ghost point, skip */
    }
  }

  /* For 3D case: nz=1, z_stride=0, nv_hnu[idx]
   * For 1D/2D: nz>1, z_stride>0, nv_hnu[nz*idx+iz]
   */
  int nz = (ndims == 3) ? 1 : (z_i + 1);

  for (int iz = 0; iz < nz; iz++) {
    int nfv = n_flow_vars + z_stride * iz;
    double n_hnu_val = (ndims == 3) ? nv_hnu[idx] : nv_hnu[nz * idx + iz];

    // Load species concentrations
    double n_O2  = u[grid_stride * idx + nfv + iO2];
    double n_O3  = u[grid_stride * idx + nfv + iO3];
    double n_1D  = u[grid_stride * idx + nfv + i1D];
    double n_1Dg = u[grid_stride * idx + nfv + i1Dg];
    double n_3Su = u[grid_stride * idx + nfv + i3Su];
    double n_1Sg = u[grid_stride * idx + nfv + i1Sg];
    double n_CO2 = u[grid_stride * idx + nfv + iCO2];

    // Compute reaction source terms
    /* O2 */
    source[grid_stride * idx + nfv + iO2] = 0.0;

    /* O3 */
    source[grid_stride * idx + nfv + iO3] =
      - (k0a + k0b) * n_hnu_val * n_O3
      - (k2a + k2b) * n_O3 * n_1D
      - (k3a + k3b) * n_O3 * n_1Sg
      - k5 * n_O3 * n_3Su
      - k6 * n_1Dg * n_O3;

    /* 1D */
    source[grid_stride * idx + nfv + i1D] =
        k0a * n_hnu_val * n_O3
      - (k1a + k1b) * n_1D * n_O2
      - (k2a + k2b) * n_1D * n_O3
      - k4 * n_1D * n_CO2;

    /* 1Dg */
    source[grid_stride * idx + nfv + i1Dg] =
        k0a * n_hnu_val * n_O3
      + k5 * n_O3 * n_3Su
      - k6 * n_1Dg * n_O3;

    /* 3Su */
    source[grid_stride * idx + nfv + i3Su] =
        k2a * n_O3 * n_1D
      - k5 * n_O3 * n_3Su;

    /* 1Sg */
    source[grid_stride * idx + nfv + i1Sg] =
        k1a * n_1D * n_O2
      - (k3a + k3b) * n_O3 * n_1Sg;

    /* CO2 */
    source[grid_stride * idx + nfv + iCO2] = 0.0;

    // Compute heating source term (energy equation)
    double Q = (
        (q0a * k0a + q0b * k0b) * n_hnu_val * n_O3
      + (q1a * k1a + q1b * k1b) * n_O2 * n_1D
      + (q2a * k2a + q2b * k2b) * n_O3 * n_1D
      + (q3a * k3a + q3b * k3b) * n_O3 * n_1Sg
      + q4 * k4 * n_1D * n_CO2
      + q5 * k5 * n_O3 * n_3Su
      + q6 * k6 * n_1Dg * n_O3
    ) * gamma_m1_inv;

    // Set energy equation source (CPU uses assignment, not addition)
    source[grid_stride * idx + n_flow_vars - 1] = Q;
  }
}

/*! Launch wrapper for GPU chemistry kernel */
extern "C" {

void gpu_launch_chemistry_source(
  double* source,
  const double* u,
  const double* nv_hnu,
  int npoints,
  int nvars,
  int n_flow_vars,
  int grid_stride,
  int z_stride,
  int z_i,
  int ndims,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  double k0a, double k0b, double k1a, double k1b,
  double k2a, double k2b, double k3a, double k3b,
  double k4, double k5, double k6,
  double q0a, double q0b, double q1a, double q1b,
  double q2a, double q2b, double q3a, double q3b,
  double q4, double q5, double q6,
  double gamma_m1_inv,
  int blockSize
)
{
#ifdef GPU_NONE
  // CPU fallback - should not be called
  fprintf(stderr, "Error: gpu_launch_chemistry_source called in CPU-only mode\n");
  exit(1);
#else
  #ifndef DEFAULT_BLOCK_SIZE
  #define DEFAULT_BLOCK_SIZE 256
  #endif

  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;

  GPU_KERNEL_LAUNCH(gpu_chemistry_source_kernel, gridSize, blockSize)(
    source, u, nv_hnu, npoints, nvars,
    n_flow_vars, grid_stride, z_stride, z_i, ndims,
    dim0, dim1, dim2, ghosts,
    k0a, k0b, k1a, k1b, k2a, k2b, k3a, k3b, k4, k5, k6,
    q0a, q0b, q1a, q1b, q2a, q2b, q3a, q3b, q4, q5, q6,
    gamma_m1_inv
  );

  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

/*! Launch wrapper for GPU photon density kernel (1D/2D cases) */
void gpu_launch_chemistry_photon_density_1d2d(
  double* nv_hnu,
  const double* u,
  const double* imap,
  int npoints,
  int grid_stride,
  int z_stride,
  int n_flow_vars,
  int nz,
  double I0,
  double c,
  double h,
  double nu,
  double n_O2_chem,
  double t_pulse_norm,
  double t_start_norm,
  double sO3,
  double dz,
  double t,
  int blockSize
)
{
#ifdef GPU_NONE
  fprintf(stderr, "Error: gpu_launch_chemistry_photon_density_1d2d called in CPU-only mode\n");
  exit(1);
#else
  #ifndef DEFAULT_BLOCK_SIZE
  #define DEFAULT_BLOCK_SIZE 256
  #endif

  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;
  int gridSize = (npoints + blockSize - 1) / blockSize;

  GPU_KERNEL_LAUNCH(gpu_chemistry_photon_density_1d2d_kernel, gridSize, blockSize)(
    nv_hnu, u, imap, npoints, grid_stride, z_stride, n_flow_vars, nz,
    I0, c, h, nu, n_O2_chem, t_pulse_norm, t_start_norm, sO3, dz, t
  );

  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

/*! Launch wrapper for GPU photon density first layer kernel (3D case) */
void gpu_launch_chemistry_photon_density_3d_first_layer(
  double* nv_hnu,
  const double* imap,
  int imax,
  int jmax,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  double I0,
  double c,
  double h,
  double nu,
  double n_O2_chem,
  double t_pulse_norm,
  double t_start_norm,
  double t,
  int blockSize
)
{
#ifdef GPU_NONE
  fprintf(stderr, "Error: gpu_launch_chemistry_photon_density_3d_first_layer called in CPU-only mode\n");
  exit(1);
#else
  #ifndef DEFAULT_BLOCK_SIZE_2D
  #define DEFAULT_BLOCK_SIZE_2D 16
  #endif

  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE_2D;

  dim3 blockDim(blockSize, blockSize);
  dim3 gridDim((imax + blockSize - 1) / blockSize, (jmax + blockSize - 1) / blockSize);

  GPU_KERNEL_LAUNCH(gpu_chemistry_photon_density_3d_first_layer_kernel, gridDim, blockDim)(
    nv_hnu, imap, imax, jmax, dim0, dim1, dim2, ghosts,
    I0, c, h, nu, n_O2_chem, t_pulse_norm, t_start_norm, t
  );

  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

/*! Launch wrapper for GPU photon density next layer kernel (3D case) */
void gpu_launch_chemistry_photon_density_3d_next_layer(
  double* nv_hnu,
  const double* u,
  int imax,
  int jmax,
  int k,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  int grid_stride,
  int n_flow_vars,
  double sO3,
  double n_O2_chem,
  double dz,
  int blockSize
)
{
#ifdef GPU_NONE
  fprintf(stderr, "Error: gpu_launch_chemistry_photon_density_3d_next_layer called in CPU-only mode\n");
  exit(1);
#else
  #ifndef DEFAULT_BLOCK_SIZE_2D
  #define DEFAULT_BLOCK_SIZE_2D 16
  #endif

  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE_2D;

  dim3 blockDim(blockSize, blockSize);
  dim3 gridDim((imax + blockSize - 1) / blockSize, (jmax + blockSize - 1) / blockSize);

  GPU_KERNEL_LAUNCH(gpu_chemistry_photon_density_3d_next_layer_kernel, gridDim, blockDim)(
    nv_hnu, u, imax, jmax, k, dim0, dim1, dim2, ghosts, grid_stride, n_flow_vars, sO3, n_O2_chem, dz
  );

  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

/*! Launch wrapper for GPU batched photon density kernel (3D case)
 *  This computes ALL z-layers in a SINGLE kernel launch, eliminating
 *  ~(kmax-1) separate launches and their associated overhead.
 */
void gpu_launch_chemistry_photon_density_3d_batched(
  double* nv_hnu,
  const double* u,
  const double* imap,
  int imax,
  int jmax,
  int kmax,
  int dim0,
  int dim1,
  int dim2,
  int ghosts,
  int grid_stride,
  int n_flow_vars,
  double I0,
  double c,
  double h,
  double nu,
  double n_O2_chem,
  double t_pulse_norm,
  double t_start_norm,
  double sO3,
  double dz,
  double t,
  int first_rank_z,
  int kstart,
  int blockSize
)
{
#ifdef GPU_NONE
  fprintf(stderr, "Error: gpu_launch_chemistry_photon_density_3d_batched called in CPU-only mode\n");
  exit(1);
#else
  #ifndef DEFAULT_BLOCK_SIZE_2D
  #define DEFAULT_BLOCK_SIZE_2D 16
  #endif

  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE_2D;

  dim3 blockDim(blockSize, blockSize);
  dim3 gridDim((imax + blockSize - 1) / blockSize, (jmax + blockSize - 1) / blockSize);

  GPU_KERNEL_LAUNCH(gpu_chemistry_photon_density_3d_batched_kernel, gridDim, blockDim)(
    nv_hnu, u, imap, imax, jmax, kmax,
    dim0, dim1, dim2, ghosts,
    grid_stride, n_flow_vars,
    I0, c, h, nu, n_O2_chem, t_pulse_norm, t_start_norm,
    sO3, dz, t, first_rank_z, kstart
  );

  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

} // extern "C"

