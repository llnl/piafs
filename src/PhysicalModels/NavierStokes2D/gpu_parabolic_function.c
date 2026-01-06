/*! @file gpu_parabolic_function.c
    @brief GPU-enabled parabolic function for 2D Navier-Stokes
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_parabolic.h>
#include <gpu_first_derivative.h>
#include <gpu_mpi.h>
#include <gpu_arrayfunctions.h>
#include <hypar.h>
#include <mpivars.h>
#include <physicalmodels/navierstokes2d.h>

int GPUNavierStokes2DParabolicFunction(double *par, double *u, void *s, void *m, double t)
{
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  NavierStokes2D *physics = (NavierStokes2D*) solver->physics;
  _DECLARE_IERR_;

  int ghosts = solver->ghosts;
  int nvars = solver->nvars;
  int *dim = solver->dim_local;
  int size = solver->npoints_local_wghosts;


  if (GPUShouldUse()) {
    GPUArraySetValue(par, 0.0, size * nvars);
    if (physics->Re <= 0) { return 0; }
    solver->count_par++;

    double inv_gamma_m1 = 1.0 / (physics->gamma - 1.0);
    double inv_Re = 1.0 / physics->Re;
    double inv_Pr = 1.0 / physics->Pr;

    /* Use persistent workspace buffers - zero allocation overhead */
    double *Q = solver->gpu_parabolic_workspace_Q;
    double *QDerivX = solver->gpu_parabolic_workspace_QDerivX;
    double *QDerivY = solver->gpu_parabolic_workspace_QDerivY;
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
    GPUMemset(FViscous, 0, size * nvars * sizeof(double));
    GPUMemset(FDeriv, 0, size * nvars * sizeof(double));

    gpu_launch_ns2d_get_primitive(Q, u, nvars, size, physics->gamma, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    solver->FirstDerivativePar(QDerivX, Q, _XDIR_, 1, solver, mpi);
    solver->FirstDerivativePar(QDerivY, Q, _YDIR_, 1, solver, mpi);
    
    GPUMPIExchangeBoundariesnD(solver->ndims, solver->nvars, dim, solver->ghosts, mpi, QDerivX);
    GPUMPIExchangeBoundariesnD(solver->ndims, solver->nvars, dim, solver->ghosts, mpi, QDerivY);
    GPUSync(); // Ensure MPI exchange is complete before proceeding

    /* Use cached metadata arrays - no allocation/copy overhead */
    int *dim_gpu = solver->gpu_dim_local;
    int *stride_gpu = solver->gpu_stride_with_ghosts;

    /* Compute dxinv offsets for each direction */
    int offset_x = 0;
    int offset_y = dim[0] + 2 * ghosts;
    
    gpu_launch_scale_array_with_dxinv(QDerivX, solver->d_dxinv, nvars, size, solver->ndims, dim_gpu, stride_gpu, ghosts, _XDIR_, offset_x, 256);
    gpu_launch_scale_array_with_dxinv(QDerivY, solver->d_dxinv, nvars, size, solver->ndims, dim_gpu, stride_gpu, ghosts, _YDIR_, offset_y, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    gpu_launch_ns2d_viscous_flux_x(FViscous, Q, QDerivX, QDerivY, nvars, size, physics->Tref, physics->T0, physics->TS, physics->TA, physics->TB, inv_Re, inv_gamma_m1, inv_Pr, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();
    solver->FirstDerivativePar(FDeriv, FViscous, _XDIR_, -1, solver, mpi);
    int npoints_interior = 1;
    for (int i = 0; i < solver->ndims; i++) npoints_interior *= dim[i];
    gpu_launch_add_scaled_derivative(par, FDeriv, solver->d_dxinv, nvars, npoints_interior, solver->ndims, dim_gpu, stride_gpu, ghosts, _XDIR_, offset_x, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    gpu_launch_ns2d_viscous_flux_y(FViscous, Q, QDerivX, QDerivY, nvars, size, physics->Tref, physics->T0, physics->TS, physics->TA, physics->TB, inv_Re, inv_gamma_m1, inv_Pr, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();
    solver->FirstDerivativePar(FDeriv, FViscous, _YDIR_, -1, solver, mpi);
    gpu_launch_add_scaled_derivative(par, FDeriv, solver->d_dxinv, nvars, npoints_interior, solver->ndims, dim_gpu, stride_gpu, ghosts, _YDIR_, offset_y, 256);
    if (GPUShouldSyncEveryOp()) GPUSync();

    /* No need to free - using persistent workspace buffers and cached metadata */
    return 0;
  } else {
    return 1; /* Fall back to CPU */
  }
}

