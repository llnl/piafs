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

  /* For 3D case, z_i is set to a large negative value, so we check ndims OR z_i < 0 */
  int nz = (ndims == 3 || z_i < 0) ? 1 : (z_i + 1);
  
  for (int iz = 0; iz < nz; iz++) {
    int nfv = n_flow_vars + z_stride * iz;
    double n_hnu_val = nv_hnu[nz * idx + iz];
    
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
    
    // Add to energy equation source
    source[grid_stride * idx + n_flow_vars - 1] += Q;
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
    k0a, k0b, k1a, k1b, k2a, k2b, k3a, k3b, k4, k5, k6,
    q0a, q0b, q1a, q1b, q2a, q2b, q3a, q3b, q4, q5, q6,
    gamma_m1_inv
  );
  
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#endif
}

} // extern "C"

