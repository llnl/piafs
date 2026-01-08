/*! @file gpu_bc_kernels.cu
    @brief GPU kernels for boundary conditions
*/

#include <gpu.h>
#include <math.h>

#ifdef GPU_CUDA
  #define GPU_KERNEL __global__
#elif defined(GPU_HIP)
  #define GPU_KERNEL __global__
#else
  #define GPU_KERNEL
#endif

/* Helper: compute 1D index from multi-dimensional index with ghosts
   (for interior cells - adds ghosts to index) */
__device__ __forceinline__ int gpu_bc_index1d(
  int ndims, const int *size, const int *index, int ghosts
)
{
  int p = index[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) {
    p = p * (size[i] + 2*ghosts) + (index[i] + ghosts);
  }
  return p;
}

/* Helper: compute 1D index with offset (for boundary cells)
   This matches _ArrayIndex1DWO_ macro in the CPU code:
   p = (index + offset + ghosts) for each dimension */
__device__ __forceinline__ int gpu_bc_index1d_wo(
  int ndims, const int *size, const int *index, const int *offset, int ghosts
)
{
  int p = index[ndims-1] + offset[ndims-1] + ghosts;
  for (int i = ndims - 2; i >= 0; i--) {
    p = p * (size[i] + 2*ghosts) + (index[i] + offset[i] + ghosts);
  }
  return p;
}

/* Kernel: Extrapolate boundary condition */
GPU_KERNEL void gpu_bc_extrapolate_kernel(
  double *phi,
  int nvars, int ndims,
  const int *size,           /* dim_local */
  int ghosts,
  int bc_dim,                /* dimension of boundary */
  int bc_face,               /* 1 = left/min, -1 = right/max */
  const int *bc_is,          /* start index of boundary zone */
  const int *bc_ie,          /* end index of boundary zone */
  int npoints                /* total boundary ghost points */
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  /* Compute bounds */
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  /* Decompose tid into multi-dimensional boundary index */
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  /* Compute interior index for extrapolation */
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  
  /* Adjust interior index based on face */
  if (bc_face == 1) {
    /* Left face: extrapolate from right */
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    /* Right face: extrapolate from left */
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }
  
  /* Compute 1D indices:
     p1 = boundary ghost cell (use offset, no ghost addition)
     p2 = interior cell (add ghosts) */
  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);
  
  /* Copy values */
  for (int v = 0; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

/* Kernel: Dirichlet boundary condition */
GPU_KERNEL void gpu_bc_dirichlet_kernel(
  double *phi,
  const double *dirichlet_value,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  /* p = boundary ghost cell (use offset, no ghost addition) */
  int p = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  
  for (int v = 0; v < nvars; v++) {
    phi[p*nvars + v] = dirichlet_value[v];
  }
}

/* Kernel: Reflect boundary condition */
GPU_KERNEL void gpu_bc_reflect_kernel(
  double *phi,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }
  
  /* p1 = boundary ghost cell (use offset), p2 = interior cell (add ghosts) */
  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);
  
  /* Copy with negation */
  for (int v = 0; v < nvars; v++) {
    phi[p1*nvars + v] = -phi[p2*nvars + v];
  }
}

/* Kernel: Periodic boundary condition (single-processor case) */
GPU_KERNEL void gpu_bc_periodic_kernel(
  double *phi,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int index1[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    index1[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  int index2[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    index2[i] = index1[i];
  }
  
  int p1, p2;
  if (bc_face == 1) {
    /* Left face: copy from right side of domain */
    index2[bc_dim] = index1[bc_dim] + size[bc_dim] - ghosts;
    /* p1 = ghost cell (use offset), p2 = interior cell (add ghosts) */
    p1 = gpu_bc_index1d_wo(ndims, size, index1, bc_is, ghosts);
    p2 = gpu_bc_index1d(ndims, size, index2, ghosts);
  } else if (bc_face == -1) {
    /* Right face: copy from left side of domain */
    /* For right face, index1 iterates within bc_is..bc_ie which are ghost cells on the right */
    /* We need to copy from the corresponding interior cells on the left */
    index2[bc_dim] = index1[bc_dim];  /* index within ghost region */
    p1 = gpu_bc_index1d_wo(ndims, size, index1, bc_is, ghosts);
    p2 = gpu_bc_index1d(ndims, size, index2, ghosts);
  } else {
    return;
  }
  
  for (int v = 0; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

/* Kernel: Slip wall BC for Euler/NS (reflect normal velocity, extrapolate others) */
GPU_KERNEL void gpu_bc_slipwall_kernel(
  double *phi,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }
  
  /* p1 = boundary ghost cell (use offset), p2 = interior cell (add ghosts) */
  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);
  
  /* For Euler/NS: rho=0, rho*u=1, rho*v=2, rho*w=3, E=4, scalars=5+ */
  /* Extrapolate all, then negate normal momentum */
  for (int v = 0; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
  /* Negate normal velocity component (momentum at index bc_dim+1) */
  int vel_idx = bc_dim + 1;
  if (vel_idx < nvars) {
    phi[p1*nvars + vel_idx] = -phi[p2*nvars + vel_idx];
  }
}

/* Kernel: No-slip wall BC for NS (moving wall, extrapolate density/pressure) */
GPU_KERNEL void gpu_bc_noslipwall_kernel(
  double *phi,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  double gamma,
  double wall_u,  /* wall velocity components */
  double wall_v,
  double wall_w,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }
  
  /* p1 = boundary ghost cell (use offset), p2 = interior cell (add ghosts) */
  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);
  
  /* Read interior values */
  double rho = phi[p2*nvars + 0];
  double rhou = phi[p2*nvars + 1];
  double rhov = (nvars >= 3) ? phi[p2*nvars + 2] : 0.0;
  double rhow = (nvars >= 4 && ndims == 3) ? phi[p2*nvars + 3] : 0.0;
  
  /* Compute velocities */
  double uvel = rhou / rho;
  double vvel = rhov / rho;
  double wvel = rhow / rho;
  
  /* Get energy (index depends on ndims: 2D->3, 3D->4) */
  int E_idx = (ndims == 2) ? 3 : 4;
  double E = phi[p2*nvars + E_idx];
  
  /* Compute pressure from interior */
  double ke = 0.5 * rho * (uvel*uvel + vvel*vvel + wvel*wvel);
  double p = (gamma - 1.0) * (E - ke);
  double inv_gamma_m1 = 1.0 / (gamma - 1.0);
  
  /* Ghost point velocities: vel_ghost = 2*wall_vel - vel_interior */
  double uvel_gpt = 2.0 * wall_u - uvel;
  double vvel_gpt = 2.0 * wall_v - vvel;
  double wvel_gpt = 2.0 * wall_w - wvel;
  
  /* Ghost point energy with ghost velocities */
  double ke_gpt = 0.5 * rho * (uvel_gpt*uvel_gpt + vvel_gpt*vvel_gpt + wvel_gpt*wvel_gpt);
  double E_gpt = inv_gamma_m1 * p + ke_gpt;
  
  /* Set ghost point values */
  phi[p1*nvars + 0] = rho;
  phi[p1*nvars + 1] = rho * uvel_gpt;
  if (ndims >= 2) phi[p1*nvars + 2] = rho * vvel_gpt;
  if (ndims == 3) phi[p1*nvars + 3] = rho * wvel_gpt;
  phi[p1*nvars + E_idx] = E_gpt;
  
  /* Copy passive scalars (if any) */
  int first_scalar = (ndims == 2) ? 4 : 5;
  for (int v = first_scalar; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

/* Kernel: Supersonic inflow (set all values) */
GPU_KERNEL void gpu_bc_supersonic_inflow_kernel(
  double *phi,
  double rho_bc, double u_bc, double v_bc, double w_bc, double p_bc,
  const double *scalars_bc,
  int n_scalars,
  double gamma,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  /* p = boundary ghost cell (use offset, no ghost addition) */
  int p = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  
  /* Set conserved variables */
  phi[p*nvars + 0] = rho_bc;
  phi[p*nvars + 1] = rho_bc * u_bc;
  if (nvars > 2) phi[p*nvars + 2] = rho_bc * v_bc;
  if (nvars > 3) phi[p*nvars + 3] = rho_bc * w_bc;
  
  double ke = 0.5 * rho_bc * (u_bc*u_bc + v_bc*v_bc + w_bc*w_bc);
  double E = p_bc / (gamma - 1.0) + ke;
  if (nvars >= 5) phi[p*nvars + 4] = E;
  
  /* Scalars */
  for (int v = 0; v < n_scalars && (5+v) < nvars; v++) {
    phi[p*nvars + 5 + v] = rho_bc * scalars_bc[v];
  }
}

/* Kernel: Supersonic outflow (extrapolate all) - same as extrapolate */
GPU_KERNEL void gpu_bc_supersonic_outflow_kernel(
  double *phi,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  /* Same as extrapolate */
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;
  
  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];
  
  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }
  
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }
  
  /* p1 = boundary ghost cell (use offset), p2 = interior cell (add ghosts) */
  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);
  
  for (int v = 0; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

/* Kernel: Subsonic inflow BC (density/velocity specified, pressure extrapolated) */
GPU_KERNEL void gpu_bc_subsonic_inflow_kernel(
  double *phi,
  double rho_bc, double u_bc, double v_bc, double w_bc,
  double gamma,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;

  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];

  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }

  /* Interior index for pressure extrapolation */
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }

  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);

  /* Get pressure from interior */
  double rho_int = phi[p2*nvars + 0];
  double rhou_int = phi[p2*nvars + 1];
  double rhov_int = (ndims >= 2) ? phi[p2*nvars + 2] : 0.0;
  double rhow_int = (ndims == 3) ? phi[p2*nvars + 3] : 0.0;
  int E_idx = (ndims == 2) ? 3 : 4;
  double E_int = phi[p2*nvars + E_idx];

  double u_int = rhou_int / rho_int;
  double v_int = rhov_int / rho_int;
  double w_int = rhow_int / rho_int;
  double ke_int = 0.5 * rho_int * (u_int*u_int + v_int*v_int + w_int*w_int);
  double p_int = (gamma - 1.0) * (E_int - ke_int);

  /* Set ghost values: specified density/velocity, extrapolated pressure */
  double inv_gamma_m1 = 1.0 / (gamma - 1.0);
  double ke_gpt = 0.5 * rho_bc * (u_bc*u_bc + v_bc*v_bc + w_bc*w_bc);
  double E_gpt = inv_gamma_m1 * p_int + ke_gpt;

  phi[p1*nvars + 0] = rho_bc;
  phi[p1*nvars + 1] = rho_bc * u_bc;
  if (ndims >= 2) phi[p1*nvars + 2] = rho_bc * v_bc;
  if (ndims == 3) phi[p1*nvars + 3] = rho_bc * w_bc;
  phi[p1*nvars + E_idx] = E_gpt;

  /* Copy passive scalars from interior */
  int first_scalar = (ndims == 2) ? 4 : 5;
  for (int v = first_scalar; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

/* Kernel: Subsonic outflow BC (pressure specified, density/velocity extrapolated) */
GPU_KERNEL void gpu_bc_subsonic_outflow_kernel(
  double *phi,
  double p_bc,
  double gamma,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;

  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];

  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }

  /* Interior index */
  int indexi[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }

  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);

  /* Get density/velocity from interior */
  double rho = phi[p2*nvars + 0];
  double rhou = phi[p2*nvars + 1];
  double rhov = (ndims >= 2) ? phi[p2*nvars + 2] : 0.0;
  double rhow = (ndims == 3) ? phi[p2*nvars + 3] : 0.0;

  double u = rhou / rho;
  double v = rhov / rho;
  double w = rhow / rho;

  /* Set ghost values: extrapolated density/velocity, specified pressure */
  double inv_gamma_m1 = 1.0 / (gamma - 1.0);
  double ke = 0.5 * rho * (u*u + v*v + w*w);
  double E_gpt = inv_gamma_m1 * p_bc + ke;

  int E_idx = (ndims == 2) ? 3 : 4;

  phi[p1*nvars + 0] = rho;
  phi[p1*nvars + 1] = rhou;
  if (ndims >= 2) phi[p1*nvars + 2] = rhov;
  if (ndims == 3) phi[p1*nvars + 3] = rhow;
  phi[p1*nvars + E_idx] = E_gpt;

  /* Copy passive scalars from interior */
  int first_scalar = (ndims == 2) ? 4 : 5;
  for (int v = first_scalar; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

/* Kernel: Subsonic ambivalent BC (decides inflow/outflow based on velocity direction) */
GPU_KERNEL void gpu_bc_subsonic_ambivalent_kernel(
  double *phi,
  double rho_bc, double u_bc, double v_bc, double w_bc, double p_bc,
  double gamma,
  int nvars, int ndims,
  const int *size,
  int ghosts,
  int bc_dim,
  int bc_face,
  const int *bc_is,
  const int *bc_ie,
  int npoints
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npoints) return;

  int bounds[3];
  for (int i = 0; i < ndims; i++) bounds[i] = bc_ie[i] - bc_is[i];

  int indexb[3] = {0,0,0};
  int tmp = tid;
  for (int i = ndims - 1; i >= 0; i--) {
    indexb[i] = tmp % bounds[i];
    tmp /= bounds[i];
  }

  /* Boundary normal (pointing into domain) */
  double nx = 0.0, ny = 0.0, nz = 0.0;
  if (bc_dim == 0) nx = 1.0;
  else if (bc_dim == 1) ny = 1.0;
  else if (bc_dim == 2) nz = 1.0;
  nx *= (double)bc_face;
  ny *= (double)bc_face;
  nz *= (double)bc_face;

  /* Get boundary face velocity (2nd order extrapolation) */
  int indexi[3] = {0,0,0};
  int indexj[3] = {0,0,0};
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
    indexj[i] = indexi[i];
  }
  if (bc_face == 1) {
    indexi[bc_dim] = 0;
    indexj[bc_dim] = 1;
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - 1;
    indexj[bc_dim] = size[bc_dim] - 2;
  }

  int pi = gpu_bc_index1d(ndims, size, indexi, ghosts);
  int pj = gpu_bc_index1d(ndims, size, indexj, ghosts);

  double rho1 = phi[pi*nvars + 0];
  double u1 = phi[pi*nvars + 1] / rho1;
  double v1 = (ndims >= 2) ? phi[pi*nvars + 2] / rho1 : 0.0;
  double w1 = (ndims == 3) ? phi[pi*nvars + 3] / rho1 : 0.0;

  double rho2 = phi[pj*nvars + 0];
  double u2 = phi[pj*nvars + 1] / rho2;
  double v2 = (ndims >= 2) ? phi[pj*nvars + 2] / rho2 : 0.0;
  double w2 = (ndims == 3) ? phi[pj*nvars + 3] / rho2 : 0.0;

  /* 2nd order extrapolation to boundary */
  double ub = 1.5*u1 - 0.5*u2;
  double vb = 1.5*v1 - 0.5*v2;
  double wb = 1.5*w1 - 0.5*w2;
  double vel_normal = ub*nx + vb*ny + wb*nz;

  /* Interior index for ghost cell */
  for (int i = 0; i < ndims; i++) {
    indexi[i] = indexb[i] + bc_is[i];
  }
  if (bc_face == 1) {
    indexi[bc_dim] = ghosts - 1 - indexb[bc_dim];
  } else if (bc_face == -1) {
    indexi[bc_dim] = size[bc_dim] - indexb[bc_dim] - 1;
  }

  int p1 = gpu_bc_index1d_wo(ndims, size, indexb, bc_is, ghosts);
  int p2 = gpu_bc_index1d(ndims, size, indexi, ghosts);

  /* Get interior values */
  double rho_int = phi[p2*nvars + 0];
  double rhou_int = phi[p2*nvars + 1];
  double rhov_int = (ndims >= 2) ? phi[p2*nvars + 2] : 0.0;
  double rhow_int = (ndims == 3) ? phi[p2*nvars + 3] : 0.0;
  int E_idx = (ndims == 2) ? 3 : 4;
  double E_int = phi[p2*nvars + E_idx];

  double u_int = rhou_int / rho_int;
  double v_int = rhov_int / rho_int;
  double w_int = rhow_int / rho_int;
  double ke_int = 0.5 * rho_int * (u_int*u_int + v_int*v_int + w_int*w_int);
  double p_int = (gamma - 1.0) * (E_int - ke_int);

  double inv_gamma_m1 = 1.0 / (gamma - 1.0);
  double rho_gpt, u_gpt, v_gpt, w_gpt, p_gpt;

  if (vel_normal > 0) {
    /* Inflow: use BC density/velocity, interior pressure */
    rho_gpt = rho_bc;
    u_gpt = u_bc;
    v_gpt = v_bc;
    w_gpt = w_bc;
    p_gpt = p_int;
  } else {
    /* Outflow: use interior density/velocity, BC pressure */
    rho_gpt = rho_int;
    u_gpt = u_int;
    v_gpt = v_int;
    w_gpt = w_int;
    p_gpt = p_bc;
  }

  double ke_gpt = 0.5 * rho_gpt * (u_gpt*u_gpt + v_gpt*v_gpt + w_gpt*w_gpt);
  double E_gpt = inv_gamma_m1 * p_gpt + ke_gpt;

  phi[p1*nvars + 0] = rho_gpt;
  phi[p1*nvars + 1] = rho_gpt * u_gpt;
  if (ndims >= 2) phi[p1*nvars + 2] = rho_gpt * v_gpt;
  if (ndims == 3) phi[p1*nvars + 3] = rho_gpt * w_gpt;
  phi[p1*nvars + E_idx] = E_gpt;

  /* Copy passive scalars from interior */
  int first_scalar = (ndims == 2) ? 4 : 5;
  for (int v = first_scalar; v < nvars; v++) {
    phi[p1*nvars + v] = phi[p2*nvars + v];
  }
}

#define DEFAULT_BLOCK_SIZE 256

/* ============================================================================
   Static device buffers for BC arrays - eliminates per-call allocation
   ============================================================================ */
static int *d_bc_size = NULL;
static int *d_bc_is = NULL;
static int *d_bc_ie = NULL;
static double *d_bc_dval = NULL;
static int d_bc_ndims_capacity = 0;
static int d_bc_nvars_capacity = 0;

static int ensure_bc_device_arrays(int ndims, int nvars_for_dval) {
  /* Check if we need to reallocate the int arrays */
  if (ndims > d_bc_ndims_capacity) {
    if (d_bc_size) { GPUFree(d_bc_size); d_bc_size = NULL; }
    if (d_bc_is) { GPUFree(d_bc_is); d_bc_is = NULL; }
    if (d_bc_ie) { GPUFree(d_bc_ie); d_bc_ie = NULL; }

    if (GPUAllocate((void**)&d_bc_size, ndims * sizeof(int))) return 1;
    if (GPUAllocate((void**)&d_bc_is, ndims * sizeof(int))) {
      GPUFree(d_bc_size); d_bc_size = NULL;
      return 1;
    }
    if (GPUAllocate((void**)&d_bc_ie, ndims * sizeof(int))) {
      GPUFree(d_bc_size); d_bc_size = NULL;
      GPUFree(d_bc_is); d_bc_is = NULL;
      return 1;
    }
    d_bc_ndims_capacity = ndims;
  }

  /* Check if we need to reallocate dval array (only if nvars_for_dval > 0) */
  if (nvars_for_dval > 0 && nvars_for_dval > d_bc_nvars_capacity) {
    if (d_bc_dval) { GPUFree(d_bc_dval); d_bc_dval = NULL; }
    if (GPUAllocate((void**)&d_bc_dval, nvars_for_dval * sizeof(double))) return 1;
    d_bc_nvars_capacity = nvars_for_dval;
  }

  return 0;
}

/* ========== Launch wrappers ========== */

extern "C" void gpu_launch_bc_extrapolate(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_extrapolate_kernel, gridSize, blockSize)(
    phi, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face, d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_dirichlet(
  double *phi, const double *dirichlet_value,
  int nvars, int ndims, const int *size, int ghosts,
  const int *bc_is, const int *bc_ie, int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, nvars)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_dval, dirichlet_value, nvars*sizeof(double));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_dirichlet_kernel, gridSize, blockSize)(
    phi, d_bc_dval, nvars, ndims, d_bc_size, ghosts, d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)dirichlet_value; (void)nvars; (void)ndims; (void)size;
  (void)ghosts; (void)bc_is; (void)bc_ie; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_reflect(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_reflect_kernel, gridSize, blockSize)(
    phi, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face, d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_periodic(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_periodic_kernel, gridSize, blockSize)(
    phi, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face, d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_slipwall(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_slipwall_kernel, gridSize, blockSize)(
    phi, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face, d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_noslipwall(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double wall_u, double wall_v, double wall_w, int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_noslipwall_kernel, gridSize, blockSize)(
    phi, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face, d_bc_is, d_bc_ie, gamma,
    wall_u, wall_v, wall_w, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie; (void)gamma;
  (void)wall_u; (void)wall_v; (void)wall_w; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_supersonic_outflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie, int blockSize
)
{
  /* Same as extrapolate */
  gpu_launch_bc_extrapolate(phi, nvars, ndims, size, ghosts, bc_dim, bc_face, bc_is, bc_ie, blockSize);
}

extern "C" void gpu_launch_bc_supersonic_inflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  const int *bc_is, const int *bc_ie,
  double gamma, double rho, double u, double v, double w, double p,
  int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_supersonic_inflow_kernel, gridSize, blockSize)(
    phi, rho, u, v, w, p, NULL, 0, gamma, nvars, ndims, d_bc_size, ghosts, d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_is; (void)bc_ie; (void)gamma; (void)rho; (void)u; (void)v; (void)w; (void)p;
  (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_subsonic_inflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double rho, double u, double v, double w,
  int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_subsonic_inflow_kernel, gridSize, blockSize)(
    phi, rho, u, v, w, gamma, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face,
    d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie;
  (void)gamma; (void)rho; (void)u; (void)v; (void)w; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_subsonic_outflow(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double p,
  int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_subsonic_outflow_kernel, gridSize, blockSize)(
    phi, p, gamma, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face,
    d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie;
  (void)gamma; (void)p; (void)blockSize;
#endif
}

extern "C" void gpu_launch_bc_subsonic_ambivalent(
  double *phi, int nvars, int ndims, const int *size, int ghosts,
  int bc_dim, int bc_face, const int *bc_is, const int *bc_ie,
  double gamma, double rho, double u, double v, double w, double p,
  int blockSize
)
{
#ifndef GPU_NONE
  if (blockSize <= 0) blockSize = DEFAULT_BLOCK_SIZE;

  int npoints = 1;
  for (int i = 0; i < ndims; i++) npoints *= (bc_ie[i] - bc_is[i]);
  if (npoints <= 0) return;

  if (ensure_bc_device_arrays(ndims, 0)) return;

  GPUCopyToDevice(d_bc_size, size, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_is, bc_is, ndims*sizeof(int));
  GPUCopyToDevice(d_bc_ie, bc_ie, ndims*sizeof(int));

  int gridSize = (npoints + blockSize - 1) / blockSize;
  GPU_KERNEL_LAUNCH(gpu_bc_subsonic_ambivalent_kernel, gridSize, blockSize)(
    phi, rho, u, v, w, p, gamma, nvars, ndims, d_bc_size, ghosts, bc_dim, bc_face,
    d_bc_is, d_bc_ie, npoints
  );
  GPU_CHECK_ERROR(GPU_GET_LAST_ERROR());
#else
  (void)phi; (void)nvars; (void)ndims; (void)size; (void)ghosts;
  (void)bc_dim; (void)bc_face; (void)bc_is; (void)bc_ie;
  (void)gamma; (void)rho; (void)u; (void)v; (void)w; (void)p; (void)blockSize;
#endif
}

