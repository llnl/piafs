/*! @file gpu_hyperbolic_function.c
    @brief GPU-enabled HyperbolicFunction implementation
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_hyperbolic.h>
#include <gpu_interpolation.h>
#include <gpu_interpolation_function.h>
#include <gpu_arrayfunctions.h>
#include <hypar.h>
#include <mpivars.h>
#include <arrayfunctions.h>
#include <interpolation.h>
#include <gpu_weno_weights.h>
#include <stdlib.h>
#include <string.h>

/* GPU-enabled HyperbolicFunction
   This version uses GPU kernels for computation when GPU is available
*/
int GPUHyperbolicFunction(
  double *hyp,           /* output: hyperbolic term */
  double *u,             /* input: solution array */
  void *s,               /* solver object */
  void *m,               /* MPI object */
  double t,              /* current time */
  int LimFlag,           /* limiter flag */
  int(*FluxFunction)(double*,double*,int,void*,double),
  int(*UpwindFunction)(double*,double*,double*,double*,double*,double*,int,void*,double)
)
{
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  int d, v, i, done;
  double *FluxI = solver->fluxI;
  double *FluxC = solver->fluxC;
  double *uL = solver->uL;
  double *uR = solver->uR;
  double *fluxL = solver->fL;
  double *fluxR = solver->fR;
  /* IMPORTANT: do not rely on IERR/CHECKERR here.
     In Release builds, CHECKERR is compiled out in this codebase, which can silently
     ignore GPU errors and lead to wrong physics (e.g., leaving WENO weights at optimal values). */

  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int size = solver->npoints_local_wghosts;
  double *x = solver->x;
  double *dxinv = GPUShouldUse() ? solver->d_dxinv : solver->dxinv;
  int index[ndims], index1[ndims], index2[ndims], dim_interface[ndims];

  LimFlag = (LimFlag && solver->flag_nonlinearinterp && solver->SetInterpLimiterVar);

  /* Initialize arrays on GPU */
  if (GPUShouldUse()) {
    GPUArraySetValue(hyp, 0.0, size * nvars);
    GPUArraySetValue(solver->StageBoundaryIntegral, 0.0, 2 * ndims * nvars);
  } else {
    _ArraySetValue_(hyp, size * nvars, 0.0);
    _ArraySetValue_(solver->StageBoundaryIntegral, 2 * ndims * nvars, 0.0);
  }

  if (!FluxFunction) return 0;
  solver->count_hyp++;

  int offset = 0;
  for (d = 0; d < ndims; d++) {
    _ArrayCopy1D_(dim, dim_interface, ndims);
    dim_interface[d]++;
    
    int size_cellcenter = 1;
    for (i = 0; i < ndims; i++) size_cellcenter *= (dim[i] + 2 * ghosts);
    
    int size_interface = 1;
    for (i = 0; i < ndims; i++) size_interface *= dim_interface[i];

    /* Evaluate cell-centered flux - FluxFunction is already set to GPU version if GPU is enabled */
    {
      int ierr_flux = FluxFunction(FluxC, u, d, solver, t);
      if (ierr_flux) return ierr_flux;
    }

    /* Compute interface fluxes using GPU if available */
    if (GPUShouldUse() && solver->InterpolateInterfacesHyp) {
      /* Check if this scheme has a GPU implementation */
      /* Note: CRWENO requires tridiagonal solve which is not GPU-accelerated,
         so it uses the CPU fallback path */
      int use_gpu_interp = 0;
      if (!strcmp(solver->spatial_scheme_hyp, _FIFTH_ORDER_WENO_) ||
          !strcmp(solver->spatial_scheme_hyp, _FIFTH_ORDER_UPWIND_) ||
          !strcmp(solver->spatial_scheme_hyp, _FOURTH_ORDER_CENTRAL_) ||
          !strcmp(solver->spatial_scheme_hyp, _SECOND_ORDER_CENTRAL_) ||
          !strcmp(solver->spatial_scheme_hyp, _FIRST_ORDER_UPWIND_) ||
          !strcmp(solver->spatial_scheme_hyp, _SECOND_ORDER_MUSCL_) ||
          !strcmp(solver->spatial_scheme_hyp, _THIRD_ORDER_MUSCL_)) {
        use_gpu_interp = 1;
      }

      /* DEBUG: Print once which path is taken */

      if (use_gpu_interp) {
        /* GPU interpolation path */
        /* Precalculate WENO nonlinear interpolation coefficients if required */
        int is_weno = !strcmp(solver->spatial_scheme_hyp, _FIFTH_ORDER_WENO_);
        if (is_weno && LimFlag && solver->SetInterpLimiterVar) {
          /* For characteristic WENO with 3D and nvars=5 or nvars=12, the fused kernel
           * computes weights inline, so skip the separate weight computation */
          int skip_weights = 0;
          if (!strcmp(solver->interp_type, _CHARACTERISTIC_) &&
              ndims == 3 && (nvars == 5 || nvars == 12)) {
            skip_weights = 1;
          }

          if (!skip_weights) {
            /* Compute WENO nonlinear weights on GPU (no host staging). */
            int ierr_w = 0;
            if (!strcmp(solver->interp_type, _CHARACTERISTIC_)) {
              ierr_w = GPUWENOFifthOrderCalculateWeightsChar(FluxC, u, d, solver, mpi);
            } else {
              ierr_w = GPUWENOFifthOrderCalculateWeights(FluxC, u, d, solver, mpi);
            }
            if (ierr_w) {
              fprintf(stderr,
                      "ERROR: GPU WENO weight computation failed (dir=%d, interp_type=%s). "
                      "This configuration is not supported on GPU yet.\n",
                      d, solver->interp_type);
              return ierr_w;
            }
          }
        }
        
        /* Call GPU interpolation through the solver function pointer.
           This ensures we honor hyp_interp_type (components vs characteristic)
           just like the CPU path does. */
        {
          int ierr_i = 0;
          ierr_i = solver->InterpolateInterfacesHyp(uL, u, u, x + offset,  1, d, solver, mpi, 1);
          if (ierr_i) return ierr_i;
          ierr_i = solver->InterpolateInterfacesHyp(uR, u, u, x + offset, -1, d, solver, mpi, 1);
          if (ierr_i) return ierr_i;
          ierr_i = solver->InterpolateInterfacesHyp(fluxL, FluxC, u, x + offset,  1, d, solver, mpi, 0);
          if (ierr_i) return ierr_i;
          ierr_i = solver->InterpolateInterfacesHyp(fluxR, FluxC, u, x + offset, -1, d, solver, mpi, 0);
          if (ierr_i) return ierr_i;
        }
        if (GPUShouldSyncEveryOp()) GPUSync();
      } else {
        /* CPU interpolation fallback for other schemes */
        /* Allocate host buffers for interpolation */
        double *u_host = (double*) malloc(size_cellcenter * nvars * sizeof(double));
        double *FluxC_host = (double*) malloc(size_cellcenter * nvars * sizeof(double));
        double *x_host = (double*) malloc((dim[d] + 2*ghosts) * sizeof(double));
        double *uL_host = (double*) malloc(size_interface * nvars * sizeof(double));
        double *uR_host = (double*) malloc(size_interface * nvars * sizeof(double));
        double *fluxL_host = (double*) malloc(size_interface * nvars * sizeof(double));
        double *fluxR_host = (double*) malloc(size_interface * nvars * sizeof(double));
        
        if (!u_host || !FluxC_host || !x_host || !uL_host || !uR_host || !fluxL_host || !fluxR_host) {
          fprintf(stderr, "Error: Failed to allocate host buffers for CPU interpolation\n");
          if (u_host) free(u_host);
          if (FluxC_host) free(FluxC_host);
          if (x_host) free(x_host);
          if (uL_host) free(uL_host);
          if (uR_host) free(uR_host);
          if (fluxL_host) free(fluxL_host);
          if (fluxR_host) free(fluxR_host);
          return 1;
        }
        
        /* Copy from GPU to host */
        GPUCopyToHost(u_host, u, size_cellcenter * nvars * sizeof(double));
        GPUCopyToHost(FluxC_host, FluxC, size_cellcenter * nvars * sizeof(double));
        GPUCopyToHost(x_host, x + offset, (dim[d] + 2*ghosts) * sizeof(double));
        if (GPUShouldSyncEveryOp()) GPUSync();
        
        /* Precalculate nonlinear interpolation coefficients if required */
        if (LimFlag && solver->SetInterpLimiterVar) {
          if (!solver->interp) {
            fprintf(stderr, "Error: GPUHyperbolicFunction: solver->interp is NULL but SetInterpLimiterVar is set\n");
            free(u_host);
            free(FluxC_host);
            free(x_host);
            free(uL_host);
            free(uR_host);
            free(fluxL_host);
            free(fluxR_host);
            return 1;
          }
          double *u_save = solver->u;
          double *x_save = solver->x;
          solver->u = u_host;
          solver->x = x_host;
          IERR solver->SetInterpLimiterVar(FluxC_host, u_host, x_host, d, solver, mpi);
          CHECKERR(ierr);
          solver->u = u_save;
          solver->x = x_save;
        }
        
        /* Call CPU interpolation functions */
        double *u_save_interp = solver->u;
        double *x_save_interp = solver->x;
        solver->u = u_host;
        solver->x = x_host;
        IERR solver->InterpolateInterfacesHyp(uL_host, u_host, u_host, x_host, 1, d, solver, mpi, 1);
        CHECKERR(ierr);
        IERR solver->InterpolateInterfacesHyp(uR_host, u_host, u_host, x_host, -1, d, solver, mpi, 1);
        CHECKERR(ierr);
        IERR solver->InterpolateInterfacesHyp(fluxL_host, FluxC_host, u_host, x_host, 1, d, solver, mpi, 0);
        CHECKERR(ierr);
        IERR solver->InterpolateInterfacesHyp(fluxR_host, FluxC_host, u_host, x_host, -1, d, solver, mpi, 0);
        CHECKERR(ierr);
        solver->u = u_save_interp;
        solver->x = x_save_interp;
        
        /* Copy back to GPU */
        GPUCopyToDevice(uL, uL_host, size_interface * nvars * sizeof(double));
        GPUCopyToDevice(uR, uR_host, size_interface * nvars * sizeof(double));
        GPUCopyToDevice(fluxL, fluxL_host, size_interface * nvars * sizeof(double));
        GPUCopyToDevice(fluxR, fluxR_host, size_interface * nvars * sizeof(double));
      if (GPUShouldSyncEveryOp()) GPUSync();
        
        free(u_host);
        free(FluxC_host);
        free(x_host);
        free(uL_host);
        free(uR_host);
        free(fluxL_host);
        free(fluxR_host);
      }

      /* Upwinding on GPU */
      if (solver->Upwind) {
        /* Use solver->Upwind which is already set to GPU version if GPU is enabled */
        int ierr_up = solver->Upwind(FluxI, fluxL, fluxR, uL, uR, u, d, solver, t);
        if (ierr_up) return ierr_up;
      } else if (UpwindFunction) {
        /* Fallback: UpwindFunction passed as parameter (should not happen if GPU is enabled) */
        /* Use default GPU upwinding */
        int ninterfaces = size_interface;
        gpu_launch_default_upwinding(FluxI, fluxL, fluxR, nvars, ninterfaces, 256);
      } else {
        /* Use GPU default upwinding */
        int ninterfaces = size_interface;
        gpu_launch_default_upwinding(FluxI, fluxL, fluxR, nvars, ninterfaces, 256);
      }
    } else if (GPUShouldUse()) {
      /* GPU enabled but InterpolateInterfacesHyp is NULL - this shouldn't happen */
      /* Copy to host, use CPU interpolation, copy back */
      fprintf(stderr, "Warning: GPU enabled but InterpolateInterfacesHyp is NULL, using CPU interpolation\n");
      
      /* Allocate host buffers */
      double *u_host = (double*) malloc(size_cellcenter * nvars * sizeof(double));
      double *FluxC_host = (double*) malloc(size_cellcenter * nvars * sizeof(double));
      double *x_host = (double*) malloc((dim[d] + 2*ghosts) * sizeof(double));
      double *FluxI_host = (double*) malloc(size_interface * nvars * sizeof(double));
      
      if (!u_host || !FluxC_host || !x_host || !FluxI_host) {
        fprintf(stderr, "Error: Failed to allocate host buffers for CPU interpolation fallback\n");
        if (u_host) free(u_host);
        if (FluxC_host) free(FluxC_host);
        if (x_host) free(x_host);
        if (FluxI_host) free(FluxI_host);
        return 1;
      }
      
      /* Copy from GPU to host */
      GPUCopyToHost(u_host, u, size_cellcenter * nvars * sizeof(double));
      GPUCopyToHost(FluxC_host, FluxC, size_cellcenter * nvars * sizeof(double));
      GPUCopyToHost(x_host, x + offset, (dim[d] + 2*ghosts) * sizeof(double));
      /* GPUCopyToHost uses a synchronous copy; avoid forced device sync here. */
      
      /* Call CPU ReconstructHyperbolic - but it's static, so we need to call HyperbolicFunction's logic */
      /* For now, use simple default upwinding on host */
      /* This is a fallback - InterpolateInterfacesHyp should always be set */
      for (int i = 0; i < size_interface; i++) {
        for (int v = 0; v < nvars; v++) {
          FluxI_host[i*nvars + v] = 0.5 * (FluxC_host[i*nvars + v] + FluxC_host[(i+1)*nvars + v]);
        }
      }
      
      /* Copy back to GPU */
      GPUCopyToDevice(FluxI, FluxI_host, size_interface * nvars * sizeof(double));
      if (GPUShouldSyncEveryOp()) GPUSync();
      
      free(u_host);
      free(FluxC_host);
      free(x_host);
      free(FluxI_host);
    } else {
      /* CPU path - should not reach here if GPU is enabled */
      return 1;
    }

    /* Calculate the first derivative using GPU kernel */
    if (GPUShouldUse()) {
      /* Use cached metadata arrays (already on device) */
      int *dim_gpu = solver->gpu_dim_local;
      int *stride_gpu = solver->gpu_stride_with_ghosts;
      
      /* Launch multi-dimensional GPU kernel */
      gpu_launch_hyperbolic_flux_derivative_nd(
        hyp, FluxI, dxinv, solver->StageBoundaryIntegral,
        nvars, ndims, dim_gpu, stride_gpu, ghosts, d, offset, 256
      );
      if (GPUShouldSyncEveryOp()) GPUSync();
    } else {
      /* CPU path */
      done = 0;
      _ArraySetValue_(index, ndims, 0);
      int p, p1, p2;

      while (!done) {
        _ArrayCopy1D_(index, index1, ndims);
        _ArrayCopy1D_(index, index2, ndims);
        index2[d]++;
        _ArrayIndex1D_(ndims, dim, index, ghosts, p);
        _ArrayIndex1D_(ndims, dim_interface, index1, 0, p1);
        _ArrayIndex1D_(ndims, dim_interface, index2, 0, p2);
        for (v = 0; v < nvars; v++) {
          hyp[nvars*p+v] += dxinv[offset+ghosts+index[d]] * (FluxI[nvars*p2+v] - FluxI[nvars*p1+v]);
        }
        /* boundary flux integral */
        if (index[d] == 0) {
          for (v = 0; v < nvars; v++) {
            solver->StageBoundaryIntegral[(2*d+0)*nvars+v] -= FluxI[nvars*p1+v];
          }
        }
        if (index[d] == dim[d] - 1) {
          for (v = 0; v < nvars; v++) {
            solver->StageBoundaryIntegral[(2*d+1)*nvars+v] += FluxI[nvars*p2+v];
          }
        }
        _ArrayIncrementIndex_(ndims, dim, index, done);
      }
    }

    offset += dim[d] + 2 * ghosts;
  }

  return 0;
}

