/*! @file gpu_bc.h
    @brief GPU boundary condition kernel declarations
*/

#ifndef _GPU_BC_H_
#define _GPU_BC_H_

#ifdef __cplusplus
extern "C" {
#endif

/* GPU BC launch wrappers */
void gpu_launch_bc_extrapolate(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
);

void gpu_launch_bc_dirichlet(
  double *phi, const double *dirichlet_value,
  int nvars, int ndims, const int *size, int ghosts,
  const int *bc_is, const int *bc_ie, int blockSize
);

void gpu_launch_bc_reflect(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
);

void gpu_launch_bc_periodic(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
);

void gpu_launch_bc_slipwall(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
);

void gpu_launch_bc_noslipwall(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double wall_u, double wall_v, double wall_w, int blockSize
);

void gpu_launch_bc_supersonic_outflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
);

void gpu_launch_bc_supersonic_inflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  const int *bc_is, const int *bc_ie,
  double gamma, double rho, double u, double v, double w, double p,
  int blockSize
);

void gpu_launch_bc_subsonic_inflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double rho, double u, double v, double w,
  int blockSize
);

void gpu_launch_bc_subsonic_outflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double p,
  int blockSize
);

void gpu_launch_bc_subsonic_ambivalent(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double rho, double u, double v, double w, double p,
  int blockSize
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_BC_H_ */

