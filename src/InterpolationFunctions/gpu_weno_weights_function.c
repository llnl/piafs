/*! @file gpu_weno_weights_function.c
    @brief Host-side wrappers for computing WENO nonlinear weights on GPU
*/

#include <stdio.h>
#include <string.h>

#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_weno_weights.h>

#include <hypar.h>
#include <mpivars.h>
#include <interpolation.h>
#include <physicalmodels/navierstokes3d.h>

/* Helper to determine the weight type from WENOParameters */
static int get_weno_weight_type(WENOParameters *weno)
{
  if (weno->yc)     return GPU_WENO_TYPE_YC;
  if (weno->borges) return GPU_WENO_TYPE_Z;
  if (weno->mapped) return GPU_WENO_TYPE_MAPPED;
  return GPU_WENO_TYPE_JS;
}

static int ensure_weno_weights_on_device(WENOParameters *weno)
{
  if (!weno) return 1;
  const size_t bytes = (size_t)(4 * weno->size) * sizeof(double);
  if (!weno->w1_gpu) { if (GPUAllocate((void**)&weno->w1_gpu, bytes)) return 1; }
  if (!weno->w2_gpu) { if (GPUAllocate((void**)&weno->w2_gpu, bytes)) return 1; }
  if (!weno->w3_gpu) { if (GPUAllocate((void**)&weno->w3_gpu, bytes)) return 1; }
  return 0;
}

int GPUWENOFifthOrderCalculateWeights(
  double *fC,
  double *uC,
  int dir,
  void *s,
  void *m
)
{
#if defined(GPU_CUDA) || defined(GPU_HIP)
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  WENOParameters *weno = (WENOParameters*) solver->interp;

  if (!GPUShouldUse()) return 1;
  if (!solver || !mpi || !weno || !fC || !uC) return 1;
  if (ensure_weno_weights_on_device(weno)) {
    if (!mpi->rank) fprintf(stderr, "Error: failed to allocate device WENO weight arrays.\n");
    return 1;
  }

  const int is_crweno = (!strcmp(solver->spatial_scheme_hyp, _FIFTH_ORDER_CRWENO_)) ? 1 : 0;
  const int weight_type = get_weno_weight_type(weno);

  gpu_launch_weno5_weights(
    fC, uC,
    weno->w1_gpu + weno->offset[dir],
    weno->w2_gpu + weno->offset[dir],
    weno->w3_gpu + weno->offset[dir],
    weno->size,
    solver->ndims, solver->nvars, solver->dim_local, solver->stride_with_ghosts,
    solver->ghosts, dir, mpi->ip[dir], mpi->iproc[dir],
    is_crweno,
    weight_type,
    weno->eps,
    256
  );
  return 0;
#else
  (void)fC; (void)uC; (void)dir; (void)s; (void)m;
  return 1;
#endif
}

int GPUWENOFifthOrderCalculateWeightsChar(
  double *fC,
  double *uC,
  int dir,
  void *s,
  void *m
)
{
#if defined(GPU_CUDA) || defined(GPU_HIP)
  HyPar *solver = (HyPar*) s;
  MPIVariables *mpi = (MPIVariables*) m;
  WENOParameters *weno = (WENOParameters*) solver->interp;

  if (!GPUShouldUse()) return 1;
  if (!solver || !mpi || !weno || !fC || !uC) {
    if (mpi && !mpi->rank) fprintf(stderr, "ERROR: GPUWENOFifthOrderCalculateWeightsChar: NULL input(s).\n");
    return 1;
  }

  /* Now supports all Euler/NS models with model-agnostic eigenvector dispatch.
     Base variables: 1D=3, 2D=4, 3D=5. nvars >= base_nvars required. */
  int base_nvars = solver->ndims + 2;
  if (solver->nvars < base_nvars) {
    if (!mpi->rank) fprintf(stderr,
                            "ERROR: GPU characteristic WENO weights require nvars >= %d for %dD (got %d).\n",
                            base_nvars, solver->ndims, solver->nvars);
    return 1;
  }

  if (ensure_weno_weights_on_device(weno)) {
    if (!mpi->rank) fprintf(stderr, "Error: failed to allocate device WENO weight arrays.\n");
    return 1;
  }

  double gamma = 1.4;
  if (solver->physics) {
    /* NavierStokes3D physics struct begins with gamma field in this codebase */
    typedef struct { double gamma; } ns3d_gamma_only;
    gamma = ((ns3d_gamma_only*)solver->physics)->gamma;
  }
  const int is_crweno = (!strcmp(solver->spatial_scheme_hyp, _FIFTH_ORDER_CRWENO_)) ? 1 : 0;
  const int weight_type = get_weno_weight_type(weno);

  gpu_launch_weno5_weights_char(
    fC, uC,
    weno->w1_gpu + weno->offset[dir],
    weno->w2_gpu + weno->offset[dir],
    weno->w3_gpu + weno->offset[dir],
    weno->size,
    solver->ndims, solver->nvars, solver->dim_local, solver->stride_with_ghosts,
    solver->ghosts, dir, mpi->ip[dir], mpi->iproc[dir],
    is_crweno,
    weight_type,
    weno->eps,
    gamma,
    256
  );
  return 0;
#else
  (void)fC; (void)uC; (void)dir; (void)s; (void)m;
  return 1;
#endif
}


