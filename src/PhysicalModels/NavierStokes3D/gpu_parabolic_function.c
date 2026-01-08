/*! @file gpu_parabolic_function.c
    @brief GPU-enabled parabolic function for 3D Navier-Stokes
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_parabolic.h>
#include <gpu_first_derivative.h>
#include <gpu_mpi.h>
#include <gpu_arrayfunctions.h>
#include <hypar.h>
#include <mpivars.h>
#include <physicalmodels/navierstokes3d.h>
#include <arrayfunctions.h>
#include <mathfunctions.h>

/* GPU-enabled 3D Navier-Stokes parabolic function */
int GPUNavierStokes3DParabolicFunction(
  double *par,
  double *u,
  void *s,
  void *m,
  double t
)
{
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  NavierStokes3D *physics = (NavierStokes3D*) solver->physics;
  _DECLARE_IERR_;

  int ghosts = solver->ghosts;
  int nvars = solver->nvars;
  int imax = solver->dim_local[0];
  int jmax = solver->dim_local[1];
  int kmax = solver->dim_local[2];
  int *dim = solver->dim_local;
  int size = solver->npoints_local_wghosts;

  if (GPUShouldUse()) {
    /* Initialize parabolic term on GPU */
    GPUArraySetValue(par, 0.0, size * nvars);

    if (physics->Re <= 0) {
      return 0; /* inviscid flow */
    }
    solver->count_par++;

    static double two_third = 2.0/3.0;
    double inv_gamma_m1 = 1.0 / (physics->gamma - 1.0);
    double inv_Re = 1.0 / physics->Re;
    double inv_Pr = 1.0 / physics->Pr;

    /* Use persistent workspace buffers - zero allocation overhead */
    double *Q = solver->gpu_parabolic_workspace_Q;
    double *QDerivX = solver->gpu_parabolic_workspace_QDerivX;
    double *QDerivY = solver->gpu_parabolic_workspace_QDerivY;
    double *QDerivZ = solver->gpu_parabolic_workspace_QDerivZ;
    double *FViscous = solver->gpu_parabolic_workspace_FViscous;
    double *FDeriv = solver->gpu_parabolic_workspace_FDeriv;
    
    /* Verify workspace is large enough */
    if (solver->gpu_parabolic_workspace_size < (size_t)(size * nvars)) {
      fprintf(stderr, "Error: Parabolic workspace too small (%zu < %d)\n",
              solver->gpu_parabolic_workspace_size, size * nvars);
      return 1;
    }
    
    /* Initialize arrays to zero */
    GPUMemset(QDerivX, 0, size * nvars * sizeof(double));
    GPUMemset(QDerivY, 0, size * nvars * sizeof(double));
    GPUMemset(QDerivZ, 0, size * nvars * sizeof(double));
    GPUMemset(FViscous, 0, size * nvars * sizeof(double));
    GPUMemset(FDeriv, 0, size * nvars * sizeof(double));

    /* Convert conserved to primitive variables on GPU */
    gpu_launch_ns3d_get_primitive(Q, u, nvars, size, physics->gamma, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* Compute derivatives using GPU */
    IERR solver->FirstDerivativePar(QDerivX, Q, _XDIR_, 1, solver, mpi);
    CHECKERR(ierr);
    IERR solver->FirstDerivativePar(QDerivY, Q, _YDIR_, 1, solver, mpi);
    CHECKERR(ierr);
    IERR solver->FirstDerivativePar(QDerivZ, Q, _ZDIR_, 1, solver, mpi);
    CHECKERR(ierr);

    /* Exchange boundaries using GPU-aware MPI */
    IERR GPUMPIExchangeBoundariesnD(_MODEL_NDIMS_, nvars, solver->dim_local,
                                      solver->ghosts, mpi, QDerivX);
    CHECKERR(ierr);
    IERR GPUMPIExchangeBoundariesnD(_MODEL_NDIMS_, nvars, solver->dim_local,
                                      solver->ghosts, mpi, QDerivY);
    CHECKERR(ierr);
    IERR GPUMPIExchangeBoundariesnD(_MODEL_NDIMS_, nvars, solver->dim_local,
                                      solver->ghosts, mpi, QDerivZ);
    CHECKERR(ierr);
    /* Note: GPUMPIExchangeBoundariesnD uses synchronous cudaMemcpy, so no explicit sync needed */

    /* Scale derivatives by dxinv, dyinv, dzinv using GPU kernels */
    /* Use cached metadata arrays - no allocation/copy overhead */
    int *dim_gpu = solver->gpu_dim_local;
    int *stride_gpu = solver->gpu_stride_with_ghosts;
    
    /* Compute offsets for each direction in dxinv */
    int offset_x = 0;
    int offset_y = dim[0] + 2 * ghosts;
    int offset_z = offset_y + dim[1] + 2 * ghosts;
    
    /* Scale derivatives using GPU kernels (use device pointer d_dxinv) */
    gpu_launch_scale_array_with_dxinv(QDerivX, solver->d_dxinv, nvars, size,
                                      _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                      ghosts, _XDIR_, offset_x, 256);
    gpu_launch_scale_array_with_dxinv(QDerivY, solver->d_dxinv, nvars, size,
                                      _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                      ghosts, _YDIR_, offset_y, 256);
    gpu_launch_scale_array_with_dxinv(QDerivZ, solver->d_dxinv, nvars, size,
                                      _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                      ghosts, _ZDIR_, offset_z, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* Compute viscous fluxes and their derivatives for X direction */
    gpu_launch_ns3d_viscous_flux_x(
      FViscous, Q, QDerivX, QDerivY, QDerivZ,
      nvars, size, physics->Tref, physics->T0, physics->TS, physics->TA, physics->TB,
      inv_Re, inv_gamma_m1, inv_Pr, 256
    );
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* Compute derivative of viscous flux in X direction */
    IERR solver->FirstDerivativePar(FDeriv, FViscous, _XDIR_, -1, solver, mpi);
    CHECKERR(ierr);

    /* Add to parabolic term using GPU kernel */
    int npoints_interior = imax * jmax * kmax;
    int offset_x_add = 0;

    gpu_launch_add_scaled_derivative(par, FDeriv, solver->d_dxinv, nvars, npoints_interior,
                                     _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                     ghosts, _XDIR_, offset_x_add, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();
    
    /* Y direction */
    gpu_launch_ns3d_viscous_flux_y(
      FViscous, Q, QDerivX, QDerivY, QDerivZ,
      nvars, size, physics->Tref, physics->T0, physics->TS, physics->TA, physics->TB,
      inv_Re, inv_gamma_m1, inv_Pr, 256
    );
    if (GPUShouldSyncEveryOp()) GPUSync();

    IERR solver->FirstDerivativePar(FDeriv, FViscous, _YDIR_, -1, solver, mpi);
    CHECKERR(ierr);
    
    int offset_y_add = dim[0] + 2 * ghosts;
    gpu_launch_add_scaled_derivative(par, FDeriv, solver->d_dxinv, nvars, npoints_interior,
                                     _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                     ghosts, _YDIR_, offset_y_add, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* Z direction */
    gpu_launch_ns3d_viscous_flux_z(
      FViscous, Q, QDerivX, QDerivY, QDerivZ,
      nvars, size, physics->Tref, physics->T0, physics->TS, physics->TA, physics->TB,
      inv_Re, inv_gamma_m1, inv_Pr, 256
    );
    if (GPUShouldSyncEveryOp()) GPUSync();

    IERR solver->FirstDerivativePar(FDeriv, FViscous, _ZDIR_, -1, solver, mpi);
    CHECKERR(ierr);
    
    int offset_z_add = offset_y + dim[1] + 2 * ghosts;
    gpu_launch_add_scaled_derivative(par, FDeriv, solver->d_dxinv, nvars, npoints_interior,
                                     _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                     ghosts, _ZDIR_, offset_z_add, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();
    
    /* No need to free - using persistent workspace buffers and cached metadata */
    return 0;
  } else {
    /* Fall back to CPU implementation */
    return 1;
  }
}

