/*! @file gpu_weno_weights.h
    @brief GPU WENO nonlinear weight computation (SetInterpLimiterVar) declarations
*/

#ifndef _GPU_WENO_WEIGHTS_H_
#define _GPU_WENO_WEIGHTS_H_

#include <interpolation.h>

/* WENO weight type constants (must match gpu_weno_weights.cu) */
#define GPU_WENO_TYPE_JS     0  /* Jiang-Shu (default) */
#define GPU_WENO_TYPE_MAPPED 1  /* Mapped WENO (Henrick et al.) */
#define GPU_WENO_TYPE_Z      2  /* WENO-Z (Borges et al.) */
#define GPU_WENO_TYPE_YC     3  /* Yamaleev-Carpenter */

#ifdef __cplusplus
extern "C" {
#endif

/* Compute WENO5 nonlinear weights on GPU into weno->w*_gpu.
   Supports all weight formulations: JS (default), mapped, WENO-Z, and YC.
   Returns 0 on success, non-zero on error. */
int GPUWENOFifthOrderCalculateWeights(
  double *fC,
  double *uC,
  int dir,
  void *solver,
  void *mpi
);

int GPUWENOFifthOrderCalculateWeightsChar(
  double *fC,
  double *uC,
  int dir,
  void *solver,
  void *mpi
);

/* Low-level launch wrappers (component-wise weights) */
void gpu_launch_weno5_weights(
  const double *fC, const double *uC,
  double *w1, double *w2, double *w3,
  int weno_size_total,
  int ndims, int nvars, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int ip_dir, int iproc_dir,
  int is_crweno,
  int weight_type,
  double eps,
  int blockSize
);

/* Low-level launch wrappers (characteristic-based weights) */
void gpu_launch_weno5_weights_char(
  const double *fC, const double *uC,
  double *w1, double *w2, double *w3,
  int weno_size_total,
  int ndims, int nvars, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int ip_dir, int iproc_dir,
  int is_crweno,
  int weight_type,
  double eps,
  double gamma,
  int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_WENO_WEIGHTS_H_ */


