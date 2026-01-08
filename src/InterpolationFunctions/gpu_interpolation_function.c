/*! @file gpu_interpolation_function.c
    @brief GPU-enabled InterpolateInterfacesHyp function
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_interpolation.h>
#include <gpu_interpolation_function.h>
#include <interpolation.h>
#include <hypar.h>
#include <interpolation.h>
#include <physicalmodels/navierstokes3d.h>

static int ensure_device_weno_weights_initialized(WENOParameters *weno)
{
  if (!weno) return 1;
  if (weno->w1_gpu && weno->w2_gpu && weno->w3_gpu) return 0;
  if (!weno->w1 || !weno->w2 || !weno->w3) {
    fprintf(stderr, "Error: WENO host weights are not allocated/initialized.\n");
    return 1;
  }
  const size_t bytes = (size_t)(4 * weno->size) * sizeof(double);
  if (!weno->w1_gpu) { if (GPUAllocate((void**)&weno->w1_gpu, bytes)) return 1; }
  if (!weno->w2_gpu) { if (GPUAllocate((void**)&weno->w2_gpu, bytes)) return 1; }
  if (!weno->w3_gpu) { if (GPUAllocate((void**)&weno->w3_gpu, bytes)) return 1; }
  /* Initialize with current host weights (at minimum, the optimal weights from WENOInitialize). */
  GPUCopyToDevice(weno->w1_gpu, weno->w1, bytes);
  GPUCopyToDevice(weno->w2_gpu, weno->w2, bytes);
  GPUCopyToDevice(weno->w3_gpu, weno->w3, bytes);
  return 0;
}

/* GPU version of InterpolateInterfacesHyp for WENO5 */
int GPUInterpolateInterfacesHypWENO5(
  double *fI,           /* output: interpolated values at interfaces */
  double *fC,           /* input: cell-centered values */
  double *u,            /* input: solution (not used for component-wise) */
  double *x,            /* input: grid coordinates (not used for uniform grid) */
  int upw,              /* upwind direction */
  int dir,              /* spatial dimension */
  void *s,              /* solver object */
  void *m,              /* MPI object */
  int uflag             /* flag: 1 for u, 0 for flux */
)
{
  HyPar *solver = (HyPar*) s;
  WENOParameters *weno = (WENOParameters*) solver->interp;
  
  if (!weno) {
    fprintf(stderr, "Error: GPUInterpolateInterfacesHypWENO5: weno is NULL\n");
    return 1;
  }
  
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  /* Compute bounds for interface array */
  int bounds_inter[3]; /* Support up to 3D */
  for (int i = 0; i < ndims; i++) {
    bounds_inter[i] = dim[i];
  }
  bounds_inter[dir] = dim[dir] + 1; /* One more interface than cells */
  
  /* Compute total interface size */
  int size_interface = 1;
  for (int i = 0; i < ndims; i++) {
    size_interface *= bounds_inter[i];
  }
  
  /* Use device-resident weights computed by GPUWENOFifthOrderCalculateWeights*. */
  if (!weno->w1_gpu || !weno->w2_gpu || !weno->w3_gpu) {
    /* Lazy init: if weights have not been computed yet this step (or nonlinear limiting disabled),
       at least ensure device weights exist by copying the initialized host weights. */
    if (ensure_device_weno_weights_initialized(weno)) {
      fprintf(stderr,
              "Error: GPUInterpolateInterfacesHypWENO5: device WENO weights are not allocated.\n");
      return 1;
    }
  }
  double *w1_gpu = weno->w1_gpu + (upw < 0 ? 2*weno->size : 0) + (uflag ? weno->size : 0) + weno->offset[dir];
  double *w2_gpu = weno->w2_gpu + (upw < 0 ? 2*weno->size : 0) + (uflag ? weno->size : 0) + weno->offset[dir];
  double *w3_gpu = weno->w3_gpu + (upw < 0 ? 2*weno->size : 0) + (uflag ? weno->size : 0) + weno->offset[dir];
  
  /* Launch GPU kernel */
  gpu_launch_weno5_interpolation_nd(
    fI, fC, w1_gpu, w2_gpu, w3_gpu,
    nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, upw, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  /* weights persist on device */
  
  return 0;
}

/* GPU version of InterpolateInterfacesHyp for WENO5 (characteristic-based) */
int GPUInterpolateInterfacesHypWENO5Char(
  double *fI,           /* output: interpolated values at interfaces */
  double *fC,           /* input: cell-centered values */
  double *u,            /* input: solution (needed for characteristic decomposition) */
  double *x,            /* input: grid coordinates (not used for uniform grid) */
  int upw,              /* upwind direction */
  int dir,              /* spatial dimension */
  void *s,              /* solver object */
  void *m,              /* MPI object */
  int uflag             /* flag: 1 for u, 0 for flux */
)
{
  HyPar *solver = (HyPar*) s;
  WENOParameters *weno = (WENOParameters*) solver->interp;

  if (!weno) {
    fprintf(stderr, "Error: GPUInterpolateInterfacesHypWENO5Char: weno is NULL\n");
    return 1;
  }

  int nvars = solver->nvars;

  /* Get physics parameters - need gamma */
  double gamma = 1.4; /* Default value */
  if (solver->physics) {
    /* Get gamma from NavierStokes3D physics structure */
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) {
      gamma = param->gamma;
    }
  }

  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;

  /* Compute bounds for interface array */
  int bounds_inter[3]; /* Support up to 3D */
  for (int i = 0; i < ndims; i++) {
    bounds_inter[i] = dim[i];
  }
  bounds_inter[dir] = dim[dir] + 1; /* One more interface than cells */

  /* Try fused WENO5 kernel for 3D with nvars=5 or nvars=12
   * This computes weights and interpolation in a single pass, avoiding
   * the need for pre-computed weights and reducing memory bandwidth */
  if (gpu_launch_weno5_fused_char_ns3d(
        fI, fC, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
        ghosts, dir, upw, weno->eps, gamma, 256) == 0) {
    if (GPUShouldSyncEveryOp()) GPUSync();
    return 0;
  }

  /* Fall back to two-pass approach: use pre-computed weights */
  if (!weno->w1_gpu || !weno->w2_gpu || !weno->w3_gpu) {
    if (ensure_device_weno_weights_initialized(weno)) {
      fprintf(stderr,
              "Error: GPUInterpolateInterfacesHypWENO5Char: device WENO weights are not allocated.\n");
      return 1;
    }
  }
  double *w1_gpu = weno->w1_gpu + (upw < 0 ? 2*weno->size : 0) + (uflag ? weno->size : 0) + weno->offset[dir];
  double *w2_gpu = weno->w2_gpu + (upw < 0 ? 2*weno->size : 0) + (uflag ? weno->size : 0) + weno->offset[dir];
  double *w3_gpu = weno->w3_gpu + (upw < 0 ? 2*weno->size : 0) + (uflag ? weno->size : 0) + weno->offset[dir];

  /* Launch GPU kernel for characteristic-based interpolation */
  gpu_launch_weno5_interpolation_nd_char(
    fI, fC, u, w1_gpu, w2_gpu, w3_gpu,
    nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, upw, gamma, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();

  /* weights persist on device */

  return 0;
}

/* Helper: map limiter_type string to limiter_id for GPU kernel */
static int get_limiter_id(const char *limiter_type)
{
  /* limiter_id mapping matches gpu_muscl2_phi in gpu_interpolation.cu:
     0: generalized minmod (gmm) - default
     1: minmod
     2: vanleer
     3: superbee */
  if (!strcmp(limiter_type, "minmod"))   return 1;
  if (!strcmp(limiter_type, "vanleer"))  return 2;
  if (!strcmp(limiter_type, "superbee")) return 3;
  return 0; /* gmm (default) */
}

/* GPU version of InterpolateInterfacesHyp for MUSCL2 (component-wise) */
int GPUInterpolateInterfacesHypMUSCL2(
  double *fI,
  double *fC,
  double *u,
  double *x,
  int upw,
  int dir,
  void *s,
  void *m,
  int uflag
)
{
  HyPar *solver = (HyPar*) s;
  MUSCLParameters *muscl = (MUSCLParameters*) solver->interp;
  
  if (!muscl) {
    fprintf(stderr, "Error: GPUInterpolateInterfacesHypMUSCL2: muscl is NULL\n");
    return 1;
  }
  
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  /* Compute bounds for interface array */
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  int limiter_id = get_limiter_id(muscl->limiter_type);
  
  gpu_launch_muscl2_interpolation_nd(
    fI, fC, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, upw, limiter_id, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  return 0;
}

/* GPU version of InterpolateInterfacesHyp for MUSCL3 (component-wise, Koren limiter form) */
int GPUInterpolateInterfacesHypMUSCL3(
  double *fI,
  double *fC,
  double *u,
  double *x,
  int upw,
  int dir,
  void *s,
  void *m,
  int uflag
)
{
  HyPar *solver = (HyPar*) s;
  MUSCLParameters *muscl = (MUSCLParameters*) solver->interp;
  
  if (!muscl) {
    fprintf(stderr, "Error: GPUInterpolateInterfacesHypMUSCL3: muscl is NULL\n");
    return 1;
  }
  
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  /* Compute bounds for interface array */
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_muscl3_interpolation_nd(
    fI, fC, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, upw, muscl->eps, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  return 0;
}

/* GPU version of InterpolateInterfacesHyp for MUSCL2 (characteristic-based) */
int GPUInterpolateInterfacesHypMUSCL2Char(
  double *fI,
  double *fC,
  double *u,
  double *x,
  int upw,
  int dir,
  void *s,
  void *m,
  int uflag
)
{
  HyPar *solver = (HyPar*) s;
  MUSCLParameters *muscl = (MUSCLParameters*) solver->interp;
  
  if (!muscl) {
    fprintf(stderr, "Error: GPUInterpolateInterfacesHypMUSCL2Char: muscl is NULL\n");
    return 1;
  }
  
  int nvars = solver->nvars;
  
  double gamma = 1.4;
  if (solver->physics) {
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) gamma = param->gamma;
  }
  
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  int limiter_id = get_limiter_id(muscl->limiter_type);
  
  gpu_launch_muscl2_interpolation_nd_char_ns3d(
    fI, fC, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, upw, limiter_id, gamma, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  return 0;
}

/* GPU version of InterpolateInterfacesHyp for MUSCL3 (characteristic-based) */
int GPUInterpolateInterfacesHypMUSCL3Char(
  double *fI,
  double *fC,
  double *u,
  double *x,
  int upw,
  int dir,
  void *s,
  void *m,
  int uflag
)
{
  HyPar *solver = (HyPar*) s;
  MUSCLParameters *muscl = (MUSCLParameters*) solver->interp;
  
  if (!muscl) {
    fprintf(stderr, "Error: GPUInterpolateInterfacesHypMUSCL3Char: muscl is NULL\n");
    return 1;
  }
  
  int nvars = solver->nvars;
  
  double gamma = 1.4;
  if (solver->physics) {
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) gamma = param->gamma;
  }
  
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_muscl3_interpolation_nd_char_ns3d(
    fI, fC, u, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
    ghosts, dir, upw, muscl->eps, gamma, 256
  );
  if (GPUShouldSyncEveryOp()) GPUSync();
  
  return 0;
}

/* ========== New interpolation scheme wrapper functions ========== */

/* GPU first order upwind (component-wise) */
int GPUInterpolateInterfacesHypFirstOrderUpwind(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_first_order_upwind_nd(fI, fC, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
                                   ghosts, dir, upw, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU first order upwind (characteristic-based) */
int GPUInterpolateInterfacesHypFirstOrderUpwindChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int nvars = solver->nvars;
  
  double gamma = 1.4;
  if (solver->physics) {
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) gamma = param->gamma;
  }
  
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_first_order_upwind_nd_char_ns3d(fI, fC, u, nvars, ndims, dim, stride_with_ghosts,
                                              bounds_inter, ghosts, dir, upw, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU second order central (component-wise) */
int GPUInterpolateInterfacesHypSecondOrderCentral(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_second_order_central_nd(fI, fC, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
                                      ghosts, dir, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU second order central (characteristic-based) */
int GPUInterpolateInterfacesHypSecondOrderCentralChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int nvars = solver->nvars;
  
  double gamma = 1.4;
  if (solver->physics) {
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) gamma = param->gamma;
  }
  
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_second_order_central_nd_char_ns3d(fI, fC, u, nvars, ndims, dim, stride_with_ghosts,
                                                bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU fourth order central (component-wise) */
int GPUInterpolateInterfacesHypFourthOrderCentral(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_fourth_order_central_nd(fI, fC, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
                                      ghosts, dir, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU fourth order central (characteristic-based) */
int GPUInterpolateInterfacesHypFourthOrderCentralChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int nvars = solver->nvars;
  
  double gamma = 1.4;
  if (solver->physics) {
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) gamma = param->gamma;
  }
  
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_fourth_order_central_nd_char_ns3d(fI, fC, u, nvars, ndims, dim, stride_with_ghosts,
                                                bounds_inter, ghosts, dir, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU fifth order upwind (component-wise) */
int GPUInterpolateInterfacesHypFifthOrderUpwind(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int ndims = solver->ndims;
  int nvars = solver->nvars;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_fifth_order_upwind_nd(fI, fC, nvars, ndims, dim, stride_with_ghosts, bounds_inter,
                                    ghosts, dir, upw, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

/* GPU fifth order upwind (characteristic-based) */
int GPUInterpolateInterfacesHypFifthOrderUpwindChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
)
{
  HyPar *solver = (HyPar*) s;
  int nvars = solver->nvars;
  
  double gamma = 1.4;
  if (solver->physics) {
    NavierStokes3D *param = (NavierStokes3D*) solver->physics;
    if (param) gamma = param->gamma;
  }
  
  int ndims = solver->ndims;
  int ghosts = solver->ghosts;
  int *dim = solver->dim_local;
  int *stride_with_ghosts = solver->stride_with_ghosts;
  
  int bounds_inter[3];
  for (int i = 0; i < ndims; i++) bounds_inter[i] = dim[i];
  bounds_inter[dir] = dim[dir] + 1;
  
  gpu_launch_fifth_order_upwind_nd_char_ns3d(fI, fC, u, nvars, ndims, dim, stride_with_ghosts,
                                              bounds_inter, ghosts, dir, upw, gamma, 256);
  if (GPUShouldSyncEveryOp()) GPUSync();
  return 0;
}

