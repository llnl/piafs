/*! @file gpu_chemistry.h
    @brief GPU chemistry function declarations
    @author Debojyoti Ghosh, Albertine Oudin
*/

#ifndef _GPU_CHEMISTRY_H_
#define _GPU_CHEMISTRY_H_

#ifdef __cplusplus
extern "C" {
#endif

/* GPU kernel launch wrapper */
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
);

/* GPU-enabled ChemistrySource function */
int GPUChemistrySource(void* a_s, double* a_U, double* a_S, 
                       void* a_p, void* a_m, double a_t);

/* GPU memory allocation for chemistry arrays */
int GPUChemistryAllocate(void* a_p, int npoints_total, int nz);

/* GPU memory deallocation for chemistry arrays */
int GPUChemistryFree(void* a_p);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_CHEMISTRY_H_ */

