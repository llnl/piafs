/*! @file gpu_chemistry.h
    @brief GPU chemistry function declarations
    @author Debojyoti Ghosh, Albertine Oudin
*/

#ifndef _GPU_CHEMISTRY_H_
#define _GPU_CHEMISTRY_H_

#ifdef __cplusplus
extern "C" {
#endif

/* GPU kernel launch wrapper for chemistry source */
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
);

/* GPU kernel launch wrapper for photon density (1D/2D) */
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
);

/* GPU kernel launch wrapper for photon density first layer (3D) */
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
);

/* GPU kernel launch wrapper for photon density next layer (3D) - LEGACY */
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
);

/* GPU kernel launch wrapper for photon density ALL layers in ONE launch (3D)
 * This is the OPTIMIZED version that eliminates ~(kmax-1) kernel launches.
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
);

/* GPU-enabled ChemistrySource function */
int GPUChemistrySource(void* a_s, double* a_U, double* a_S,
                       void* a_p, void* a_m, double a_t);

/* GPU-enabled ChemistrySetPhotonDensity function */
int GPUChemistrySetPhotonDensity(void* a_s, void* a_p, void* a_m,
                                  double* a_U, double a_t);

/* GPU memory allocation for chemistry arrays */
int GPUChemistryAllocate(void* a_p, int npoints_total, int nz);

/* GPU memory deallocation for chemistry arrays */
int GPUChemistryFree(void* a_p);

/* Copy nv_hnu from GPU to host (for output/diagnostics) */
int GPUChemistryCopyPhotonDensityToHost(void* a_p);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_CHEMISTRY_H_ */

