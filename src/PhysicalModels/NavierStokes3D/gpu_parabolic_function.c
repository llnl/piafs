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

    /* Allocate primitive variables and derivatives on GPU */
    double *Q, *QDerivX, *QDerivY, *QDerivZ, *FViscous, *FDeriv;
    
    if (GPUAllocate((void**)&Q, size * nvars * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate Q on GPU\n");
      return 1;
    }
    if (GPUAllocate((void**)&QDerivX, size * nvars * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate QDerivX on GPU\n");
      GPUFree(Q);
      return 1;
    }
    if (GPUAllocate((void**)&QDerivY, size * nvars * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate QDerivY on GPU\n");
      GPUFree(Q);
      GPUFree(QDerivX);
      return 1;
    }
    if (GPUAllocate((void**)&QDerivZ, size * nvars * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate QDerivZ on GPU\n");
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      return 1;
    }
    if (GPUAllocate((void**)&FViscous, size * nvars * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate FViscous on GPU\n");
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      GPUFree(QDerivZ);
      return 1;
    }
    if (GPUAllocate((void**)&FDeriv, size * nvars * sizeof(double))) {
      fprintf(stderr, "Error: Failed to allocate FDeriv on GPU\n");
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      GPUFree(QDerivZ);
      GPUFree(FViscous);
      return 1;
    }

    /* Convert conserved to primitive variables on GPU */
    gpu_launch_ns3d_get_primitive(Q, u, nvars, size, physics->gamma, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* Compute derivatives using GPU */
    IERR GPUFirstDerivativeSecondOrderCentral(QDerivX, Q, _XDIR_, 1, solver, mpi);
    CHECKERR(ierr);
    IERR GPUFirstDerivativeSecondOrderCentral(QDerivY, Q, _YDIR_, 1, solver, mpi);
    CHECKERR(ierr);
    IERR GPUFirstDerivativeSecondOrderCentral(QDerivZ, Q, _ZDIR_, 1, solver, mpi);
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

    /* Scale derivatives by dxinv, dyinv, dzinv using GPU kernels */
    /* Allocate GPU arrays for dim and stride_with_ghosts */
    int *dim_gpu = NULL;
    int *stride_gpu = NULL;
    
    if (GPUAllocate((void**)&dim_gpu, _MODEL_NDIMS_ * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate dim_gpu\n");
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      GPUFree(QDerivZ);
      GPUFree(FViscous);
      GPUFree(FDeriv);
      return 1;
    }
    if (GPUAllocate((void**)&stride_gpu, _MODEL_NDIMS_ * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate stride_gpu\n");
      GPUFree(dim_gpu);
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      GPUFree(QDerivZ);
      GPUFree(FViscous);
      GPUFree(FDeriv);
      return 1;
    }
    
    GPUCopyToDevice(dim_gpu, dim, _MODEL_NDIMS_ * sizeof(int));
    GPUCopyToDevice(stride_gpu, solver->stride_with_ghosts, _MODEL_NDIMS_ * sizeof(int));
    if (GPUShouldSyncEveryOp()) GPUSync();
    
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
    
    GPUFree(dim_gpu);
    GPUFree(stride_gpu);

    /* Compute viscous fluxes using GPU kernels */
    /* For X direction */
    gpu_launch_ns3d_viscous_flux_x(
      FViscous, Q, QDerivX, QDerivY, QDerivZ,
      nvars, size, physics->Tref, physics->T0, physics->TS, physics->TA, physics->TB,
      inv_Re, inv_gamma_m1, inv_Pr, 256
    );
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* Compute derivative of viscous flux */
    IERR GPUFirstDerivativeSecondOrderCentral(FDeriv, FViscous, _XDIR_, -1, solver, mpi);
    CHECKERR(ierr);

    /* Add to parabolic term using GPU kernel */
    int npoints_interior = imax * jmax * kmax;
    int offset_x_add = 0;
    
    /* Reallocate GPU arrays for dim and stride_with_ghosts (were freed after scaling) */
    dim_gpu = NULL;
    stride_gpu = NULL;
    
    if (GPUAllocate((void**)&dim_gpu, _MODEL_NDIMS_ * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate dim_gpu for adding scaled derivative\n");
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      GPUFree(QDerivZ);
      GPUFree(FViscous);
      GPUFree(FDeriv);
      return 1;
    }
    if (GPUAllocate((void**)&stride_gpu, _MODEL_NDIMS_ * sizeof(int))) {
      fprintf(stderr, "Error: Failed to allocate stride_gpu for adding scaled derivative\n");
      GPUFree(dim_gpu);
      GPUFree(Q);
      GPUFree(QDerivX);
      GPUFree(QDerivY);
      GPUFree(QDerivZ);
      GPUFree(FViscous);
      GPUFree(FDeriv);
      return 1;
    }
    
    GPUCopyToDevice(dim_gpu, dim, _MODEL_NDIMS_ * sizeof(int));
    GPUCopyToDevice(stride_gpu, solver->stride_with_ghosts, _MODEL_NDIMS_ * sizeof(int));
    if (GPUShouldSyncEveryOp()) GPUSync();
    
    gpu_launch_add_scaled_derivative(par, FDeriv, solver->d_dxinv, nvars, npoints_interior,
                                     _MODEL_NDIMS_, dim_gpu, stride_gpu,
                                     ghosts, _XDIR_, offset_x_add, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();
    
    GPUFree(dim_gpu);
    GPUFree(stride_gpu);

    /* Similar for Y and Z directions */
    /* TODO: Complete Y and Z direction computation */

    /* Free GPU memory */
    GPUFree(Q);
    GPUFree(QDerivX);
    GPUFree(QDerivY);
    GPUFree(QDerivZ);
    GPUFree(FViscous);
    GPUFree(FDeriv);

    if (GPUShouldSyncEveryOp()) GPUSync();
    return 0;
  } else {
    /* Fall back to CPU implementation */
    return 1;
  }
}

