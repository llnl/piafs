/*! @file gpu_weno_weights.cu
    @brief GPU kernels for WENO5 nonlinear weight computation (Jiang-Shu)
*/

#include <gpu.h>
#include <gpu_launch.h>

#include <physicalmodels/gpu_euler1d_helpers.h>
#include <physicalmodels/gpu_ns2d_helpers.h>
#include <physicalmodels/gpu_ns3d_helpers.h>

/* Unified Roe average dispatch based on ndims */
static __device__ __forceinline__ void gpu_roe_average_weno(
  double *uavg, const double *uL, const double *uR,
  int base_nvars, int ndims, double gamma
) {
  if (ndims == 1) {
    gpu_euler1d_roe_average(uavg, uL, uR, base_nvars, gamma);
  } else if (ndims == 2) {
    gpu_ns2d_roe_average(uavg, uL, uR, base_nvars, gamma);
  } else {
    gpu_ns3d_roe_average(uavg, uL, uR, base_nvars, gamma);
  }
}

/* Unified left eigenvector dispatch based on ndims */
static __device__ __forceinline__ void gpu_left_eigenvectors_weno(
  const double *u, double *L, double gamma, int base_nvars, int ndims, int dir
) {
  if (ndims == 1) {
    gpu_euler1d_left_eigenvectors(u, L, gamma, base_nvars);
  } else if (ndims == 2) {
    gpu_ns2d_left_eigenvectors(u, L, gamma, base_nvars, dir);
  } else {
    gpu_ns3d_left_eigenvectors(u, L, gamma, base_nvars, dir);
  }
}

/* WENO weight type constants */
#define WENO_TYPE_JS     0  /* Jiang-Shu (default) */
#define WENO_TYPE_MAPPED 1  /* Mapped WENO (Henrick et al.) */
#define WENO_TYPE_Z      2  /* WENO-Z (Borges et al.) */
#define WENO_TYPE_YC     3  /* Yamaleev-Carpenter */

/* Compute smoothness indicators (common to all WENO weight formulations) */
static __device__ __forceinline__ void weno_smoothness_indicators(
  double m3, double m2, double m1, double p1, double p2,
  double *b1, double *b2, double *b3
)
{
  const double thirteen_by_twelve = 13.0/12.0;
  const double one_fourth = 1.0/4.0;

  *b1 = thirteen_by_twelve*(m3 - 2*m2 + m1)*(m3 - 2*m2 + m1)
      + one_fourth*(m3 - 4*m2 + 3*m1)*(m3 - 4*m2 + 3*m1);
  *b2 = thirteen_by_twelve*(m2 - 2*m1 + p1)*(m2 - 2*m1 + p1)
      + one_fourth*(m2 - p1)*(m2 - p1);
  *b3 = thirteen_by_twelve*(m1 - 2*p1 + p2)*(m1 - 2*p1 + p2)
      + one_fourth*(3*m1 - 4*p1 + p2)*(3*m1 - 4*p1 + p2);
}

/* Jiang-Shu (default) WENO weights */
static __device__ __forceinline__ void weno_js_weights_5pt(
  double m3, double m2, double m1, double p1, double p2,
  double c1, double c2, double c3, double eps,
  double *w1, double *w2, double *w3
)
{
  double b1, b2, b3;
  weno_smoothness_indicators(m3, m2, m1, p1, p2, &b1, &b2, &b3);

  const double a1 = c1 / ((b1 + eps) * (b1 + eps));
  const double a2 = c2 / ((b2 + eps) * (b2 + eps));
  const double a3 = c3 / ((b3 + eps) * (b3 + eps));

  const double inv = 1.0 / (a1 + a2 + a3);
  *w1 = a1 * inv;
  *w2 = a2 * inv;
  *w3 = a3 * inv;
}

/* Mapped WENO weights (Henrick, Aslam, Powers) */
static __device__ __forceinline__ void weno_mapped_weights_5pt(
  double m3, double m2, double m1, double p1, double p2,
  double c1, double c2, double c3, double eps,
  double *w1, double *w2, double *w3
)
{
  double b1, b2, b3;
  weno_smoothness_indicators(m3, m2, m1, p1, p2, &b1, &b2, &b3);

  /* First compute JS weights (tilde_omega) */
  double a1 = c1 / ((b1 + eps) * (b1 + eps));
  double a2 = c2 / ((b2 + eps) * (b2 + eps));
  double a3 = c3 / ((b3 + eps) * (b3 + eps));

  double inv = 1.0 / (a1 + a2 + a3);
  double tw1 = a1 * inv;
  double tw2 = a2 * inv;
  double tw3 = a3 * inv;

  /* Apply mapping function: a = tilde_omega * (c + c^2 - 3*c*tilde_omega + tilde_omega^2) / (c^2 + tilde_omega*(1 - 2*c)) */
  a1 = tw1 * (c1 + c1*c1 - 3*c1*tw1 + tw1*tw1) / (c1*c1 + tw1*(1.0 - 2.0*c1));
  a2 = tw2 * (c2 + c2*c2 - 3*c2*tw2 + tw2*tw2) / (c2*c2 + tw2*(1.0 - 2.0*c2));
  a3 = tw3 * (c3 + c3*c3 - 3*c3*tw3 + tw3*tw3) / (c3*c3 + tw3*(1.0 - 2.0*c3));

  inv = 1.0 / (a1 + a2 + a3);
  *w1 = a1 * inv;
  *w2 = a2 * inv;
  *w3 = a3 * inv;
}

/* WENO-Z weights (Borges et al.) */
static __device__ __forceinline__ void weno_z_weights_5pt(
  double m3, double m2, double m1, double p1, double p2,
  double c1, double c2, double c3, double eps,
  double *w1, double *w2, double *w3
)
{
  double b1, b2, b3;
  weno_smoothness_indicators(m3, m2, m1, p1, p2, &b1, &b2, &b3);

  /* tau = |b1 - b3| */
  double tau = fabs(b1 - b3);

  /* a_k = c_k * (1 + (tau/(b_k + eps))^2) */
  const double a1 = c1 * (1.0 + (tau/(b1+eps)) * (tau/(b1+eps)));
  const double a2 = c2 * (1.0 + (tau/(b2+eps)) * (tau/(b2+eps)));
  const double a3 = c3 * (1.0 + (tau/(b3+eps)) * (tau/(b3+eps)));

  const double inv = 1.0 / (a1 + a2 + a3);
  *w1 = a1 * inv;
  *w2 = a2 * inv;
  *w3 = a3 * inv;
}

/* Yamaleev-Carpenter (YC/ESWENO) weights */
static __device__ __forceinline__ void weno_yc_weights_5pt(
  double m3, double m2, double m1, double p1, double p2,
  double c1, double c2, double c3, double eps,
  double *w1, double *w2, double *w3
)
{
  double b1, b2, b3;
  weno_smoothness_indicators(m3, m2, m1, p1, p2, &b1, &b2, &b3);

  /* tau = (m3 - 4*m2 + 6*m1 - 4*p1 + p2)^2 */
  double tau_val = m3 - 4*m2 + 6*m1 - 4*p1 + p2;
  double tau = tau_val * tau_val;

  /* a_k = c_k * (1 + (tau/(b_k + eps))^2) */
  const double a1 = c1 * (1.0 + (tau/(b1+eps)) * (tau/(b1+eps)));
  const double a2 = c2 * (1.0 + (tau/(b2+eps)) * (tau/(b2+eps)));
  const double a3 = c3 * (1.0 + (tau/(b3+eps)) * (tau/(b3+eps)));

  const double inv = 1.0 / (a1 + a2 + a3);
  *w1 = a1 * inv;
  *w2 = a2 * inv;
  *w3 = a3 * inv;
}

/* Unified weight computation dispatcher */
static __device__ __forceinline__ void weno_weights_5pt(
  double m3, double m2, double m1, double p1, double p2,
  double c1, double c2, double c3, double eps,
  int weight_type,
  double *w1, double *w2, double *w3
)
{
  switch (weight_type) {
    case WENO_TYPE_MAPPED:
      weno_mapped_weights_5pt(m3, m2, m1, p1, p2, c1, c2, c3, eps, w1, w2, w3);
      break;
    case WENO_TYPE_Z:
      weno_z_weights_5pt(m3, m2, m1, p1, p2, c1, c2, c3, eps, w1, w2, w3);
      break;
    case WENO_TYPE_YC:
      weno_yc_weights_5pt(m3, m2, m1, p1, p2, c1, c2, c3, eps, w1, w2, w3);
      break;
    default: /* WENO_TYPE_JS */
      weno_js_weights_5pt(m3, m2, m1, p1, p2, c1, c2, c3, eps, w1, w2, w3);
      break;
  }
}

__global__ void gpu_weno5_weights_kernel(
  const double *fC, const double *uC,
  double *w1_base, double *w2_base, double *w3_base,
  int weno_size_total,
  int ndims, int nvars,
  int dim0, int dim1, int dim2,
  int ghosts,
  int dir,
  int stride_dir,
  int ip_dir, int iproc_dir,
  int is_crweno,
  int weight_type,
  double eps
)
{
  /* bounds_inter: dim[dir]+1, others dim */
  const int bi0 = (dir == 0 ? (dim0 + 1) : dim0);
  const int bi1 = (ndims > 1 ? (dir == 1 ? (dim1 + 1) : dim1) : 1);
  const int bi2 = (ndims > 2 ? (dir == 2 ? (dim2 + 1) : dim2) : 1);
  const int ninterfaces = bi0 * bi1 * bi2;

  int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ninterfaces) return;

  int i0 = 0, i1 = 0, i2 = 0;
  if (ndims == 1) {
    i0 = p;
  } else if (ndims == 2) {
    i0 = p % bi0;
    i1 = p / bi0;
  } else {
    int t = p;
    i0 = t % bi0; t /= bi0;
    i1 = t % bi1; t /= bi1;
    i2 = t;
  }

  const int idxI_dir = (dir == 0 ? i0 : (dir == 1 ? i1 : i2)); /* 0..dim[dir] */

  /* optimal weights */
  double c1 = 0.1, c2 = 0.6, c3 = 0.3;
  if (is_crweno) {
    /* match CPU: use WENO5 weights at the "physical" boundaries, CRWENO5 interior */
    if (   ((ip_dir == 0) && (idxI_dir == 0))
        || ((ip_dir == (iproc_dir - 1)) && (idxI_dir == (dir == 0 ? dim0 : (dir == 1 ? dim1 : dim2))))) {
      c1 = 0.1; c2 = 0.6; c3 = 0.3;
    } else {
      c1 = 0.2; c2 = 0.5; c3 = 0.3;
    }
  }

  /* compute qm1L and qm1R (cell indices include ghosts) */
  int c0L = i0, c1L = i1, c2L = i2;
  int c0R = i0, c1R = i1, c2R = i2;
  if (dir == 0) { c0L = i0 - 1; c0R = i0; }
  else if (dir == 1) { c1L = i1 - 1; c1R = i1; }
  else { c2L = i2 - 1; c2R = i2; }

  const int sx1 = (dim0 + 2*ghosts);
  const int sx2 = (dim0 + 2*ghosts) * (dim1 + 2*ghosts);

  const int qm1L = (ndims == 1)
    ? (c0L + ghosts)
    : (ndims == 2)
      ? ((c1L + ghosts) * sx1 + (c0L + ghosts))
      : ((c2L + ghosts) * sx2 + (c1L + ghosts) * sx1 + (c0L + ghosts));

  const int qm1R = (ndims == 1)
    ? (c0R + ghosts)
    : (ndims == 2)
      ? ((c1R + ghosts) * sx1 + (c0R + ghosts))
      : ((c2R + ghosts) * sx2 + (c1R + ghosts) * sx1 + (c0R + ghosts));

  const int qm3L = qm1L - 2*stride_dir;
  const int qm2L = qm1L -   stride_dir;
  const int qp1L = qm1L +   stride_dir;
  const int qp2L = qm1L + 2*stride_dir;

  const int qm3R = qm1R + 2*stride_dir;
  const int qm2R = qm1R +   stride_dir;
  const int qp1R = qm1R -   stride_dir;
  const int qp2R = qm1R - 2*stride_dir;

  /* layout matches WENOInitialize:
     base pointers correspond to (LF) block; (RF) is +2*size; (LU) is +size; (RU) is +3*size.
     Here, w*_base already points to offset[dir] into the LF block.
  */
  /* WENOParameters layout: blocks of length weno->size (total across dirs):
     [LF | LU | RF | RU]. w*_base points to (LF + offset[dir]). */
  const int size_block = weno_size_total;
  double *w1LF = w1_base;
  double *w2LF = w2_base;
  double *w3LF = w3_base;
  double *w1RF = w1_base + 2*size_block;
  double *w2RF = w2_base + 2*size_block;
  double *w3RF = w3_base + 2*size_block;
  double *w1LU = w1_base + 1*size_block;
  double *w2LU = w2_base + 1*size_block;
  double *w3LU = w3_base + 1*size_block;
  double *w1RU = w1_base + 3*size_block;
  double *w2RU = w2_base + 3*size_block;
  double *w3RU = w3_base + 3*size_block;

  for (int v = 0; v < nvars; v++) {
    double w1, w2, w3;
    /* flux weights */
    weno_weights_5pt(
      fC[qm3L*nvars + v], fC[qm2L*nvars + v], fC[qm1L*nvars + v], fC[qp1L*nvars + v], fC[qp2L*nvars + v],
      c1, c2, c3, eps, weight_type, &w1, &w2, &w3
    );
    w1LF[p*nvars + v] = w1; w2LF[p*nvars + v] = w2; w3LF[p*nvars + v] = w3;

    weno_weights_5pt(
      fC[qm3R*nvars + v], fC[qm2R*nvars + v], fC[qm1R*nvars + v], fC[qp1R*nvars + v], fC[qp2R*nvars + v],
      c1, c2, c3, eps, weight_type, &w1, &w2, &w3
    );
    w1RF[p*nvars + v] = w1; w2RF[p*nvars + v] = w2; w3RF[p*nvars + v] = w3;

    /* solution weights */
    weno_weights_5pt(
      uC[qm3L*nvars + v], uC[qm2L*nvars + v], uC[qm1L*nvars + v], uC[qp1L*nvars + v], uC[qp2L*nvars + v],
      c1, c2, c3, eps, weight_type, &w1, &w2, &w3
    );
    w1LU[p*nvars + v] = w1; w2LU[p*nvars + v] = w2; w3LU[p*nvars + v] = w3;

    weno_weights_5pt(
      uC[qm3R*nvars + v], uC[qm2R*nvars + v], uC[qm1R*nvars + v], uC[qp1R*nvars + v], uC[qp2R*nvars + v],
      c1, c2, c3, eps, weight_type, &w1, &w2, &w3
    );
    w1RU[p*nvars + v] = w1; w2RU[p*nvars + v] = w2; w3RU[p*nvars + v] = w3;
  }
}

__global__ void gpu_weno5_weights_char_kernel(
  const double *fC, const double *uC,
  double *w1_base, double *w2_base, double *w3_base,
  int weno_size_total,
  int ndims, int nvars,
  int dim0, int dim1, int dim2,
  int ghosts,
  int dir,
  int stride_dir,
  int ip_dir, int iproc_dir,
  int is_crweno,
  int weight_type,
  double eps,
  double gamma
)
{
  const int bi0 = (dir == 0 ? (dim0 + 1) : dim0);
  const int bi1 = (ndims > 1 ? (dir == 1 ? (dim1 + 1) : dim1) : 1);
  const int bi2 = (ndims > 2 ? (dir == 2 ? (dim2 + 1) : dim2) : 1);
  const int ninterfaces = bi0 * bi1 * bi2;

  int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ninterfaces) return;

  int i0 = 0, i1 = 0, i2 = 0;
  if (ndims == 1) {
    i0 = p;
  } else if (ndims == 2) {
    i0 = p % bi0;
    i1 = p / bi0;
  } else {
    int t = p;
    i0 = t % bi0; t /= bi0;
    i1 = t % bi1; t /= bi1;
    i2 = t;
  }

  const int idxI_dir = (dir == 0 ? i0 : (dir == 1 ? i1 : i2));
  double c1 = 0.1, c2 = 0.6, c3 = 0.3;
  if (is_crweno) {
    if (   ((ip_dir == 0) && (idxI_dir == 0))
        || ((ip_dir == (iproc_dir - 1)) && (idxI_dir == (dir == 0 ? dim0 : (dir == 1 ? dim1 : dim2))))) {
      c1 = 0.1; c2 = 0.6; c3 = 0.3;
    } else {
      c1 = 0.2; c2 = 0.5; c3 = 0.3;
    }
  }

  int c0L = i0, c1L = i1, c2L = i2;
  int c0R = i0, c1R = i1, c2R = i2;
  if (dir == 0) { c0L = i0 - 1; c0R = i0; }
  else if (dir == 1) { c1L = i1 - 1; c1R = i1; }
  else { c2L = i2 - 1; c2R = i2; }

  const int sx1 = (dim0 + 2*ghosts);
  const int sx2 = (dim0 + 2*ghosts) * (dim1 + 2*ghosts);

  const int qm1L = (ndims == 1)
    ? (c0L + ghosts)
    : (ndims == 2)
      ? ((c1L + ghosts) * sx1 + (c0L + ghosts))
      : ((c2L + ghosts) * sx2 + (c1L + ghosts) * sx1 + (c0L + ghosts));

  const int qm1R = (ndims == 1)
    ? (c0R + ghosts)
    : (ndims == 2)
      ? ((c1R + ghosts) * sx1 + (c0R + ghosts))
      : ((c2R + ghosts) * sx2 + (c1R + ghosts) * sx1 + (c0R + ghosts));

  const int qm3L = qm1L - 2*stride_dir;
  const int qm2L = qm1L -   stride_dir;
  const int qp1L = qm1L +   stride_dir;
  const int qp2L = qm1L + 2*stride_dir;

  const int qm3R = qm1R + 2*stride_dir;
  const int qm2R = qm1R +   stride_dir;
  const int qp1R = qm1R -   stride_dir;
  const int qp2R = qm1R - 2*stride_dir;

  /* Determine base number of variables from ndims: 1D=3, 2D=4, 3D=5 */
  const int base_nvars = ndims + 2;

  /* averaged state and left eigenvectors at this interface (match CPU):
     - eigenvectors depend only on the base Euler/NS variables
     - passive scalars (if any) are identity in characteristic space */
  double uavg[5];  /* max base_nvars = 5 */
  gpu_roe_average_weno(uavg, uC + qm1L*nvars, uC + qp1L*nvars, base_nvars, ndims, gamma);
  double Lmat[25];  /* max 5x5 */
  gpu_left_eigenvectors_weno(uavg, Lmat, gamma, base_nvars, ndims, dir);

  const int size_block = weno_size_total;
  double *w1LF = w1_base;
  double *w2LF = w2_base;
  double *w3LF = w3_base;
  double *w1RF = w1_base + 2*size_block;
  double *w2RF = w2_base + 2*size_block;
  double *w3RF = w3_base + 2*size_block;
  double *w1LU = w1_base + 1*size_block;
  double *w2LU = w2_base + 1*size_block;
  double *w3LU = w3_base + 1*size_block;
  double *w1RU = w1_base + 3*size_block;
  double *w2RU = w2_base + 3*size_block;
  double *w3RU = w3_base + 3*size_block;

  for (int v = 0; v < nvars; v++) {
    /* Characteristic stencil values (scalar) for this characteristic field */
    double fm3LF, fm2LF, fm1LF, fp1LF, fp2LF;
    double fm3RF, fm2RF, fm1RF, fp1RF, fp2RF;
    double um3LF, um2LF, um1LF, up1LF, up2LF;
    double um3RF, um2RF, um1RF, up1RF, up2RF;

    if (v < base_nvars) {
      /* Use eigenvectors for the base Euler/NS subsystem */
      const double *Lv = &Lmat[v*base_nvars];
      fm3LF = fm2LF = fm1LF = fp1LF = fp2LF = 0.0;
      fm3RF = fm2RF = fm1RF = fp1RF = fp2RF = 0.0;
      um3LF = um2LF = um1LF = up1LF = up2LF = 0.0;
      um3RF = um2RF = um1RF = up1RF = up2RF = 0.0;
      for (int k = 0; k < base_nvars; k++) {
        const double Lvk = Lv[k];
        fm3LF += Lvk * fC[qm3L*nvars + k];
        fm2LF += Lvk * fC[qm2L*nvars + k];
        fm1LF += Lvk * fC[qm1L*nvars + k];
        fp1LF += Lvk * fC[qp1L*nvars + k];
        fp2LF += Lvk * fC[qp2L*nvars + k];

        fm3RF += Lvk * fC[qm3R*nvars + k];
        fm2RF += Lvk * fC[qm2R*nvars + k];
        fm1RF += Lvk * fC[qm1R*nvars + k];
        fp1RF += Lvk * fC[qp1R*nvars + k];
        fp2RF += Lvk * fC[qp2R*nvars + k];

        um3LF += Lvk * uC[qm3L*nvars + k];
        um2LF += Lvk * uC[qm2L*nvars + k];
        um1LF += Lvk * uC[qm1L*nvars + k];
        up1LF += Lvk * uC[qp1L*nvars + k];
        up2LF += Lvk * uC[qp2L*nvars + k];

        um3RF += Lvk * uC[qm3R*nvars + k];
        um2RF += Lvk * uC[qm2R*nvars + k];
        um1RF += Lvk * uC[qm1R*nvars + k];
        up1RF += Lvk * uC[qp1R*nvars + k];
        up2RF += Lvk * uC[qp2R*nvars + k];
      }
    } else {
      /* Passive scalars: characteristic transform is identity */
      fm3LF = fC[qm3L*nvars + v];
      fm2LF = fC[qm2L*nvars + v];
      fm1LF = fC[qm1L*nvars + v];
      fp1LF = fC[qp1L*nvars + v];
      fp2LF = fC[qp2L*nvars + v];

      fm3RF = fC[qm3R*nvars + v];
      fm2RF = fC[qm2R*nvars + v];
      fm1RF = fC[qm1R*nvars + v];
      fp1RF = fC[qp1R*nvars + v];
      fp2RF = fC[qp2R*nvars + v];

      um3LF = uC[qm3L*nvars + v];
      um2LF = uC[qm2L*nvars + v];
      um1LF = uC[qm1L*nvars + v];
      up1LF = uC[qp1L*nvars + v];
      up2LF = uC[qp2L*nvars + v];

      um3RF = uC[qm3R*nvars + v];
      um2RF = uC[qm2R*nvars + v];
      um1RF = uC[qm1R*nvars + v];
      up1RF = uC[qp1R*nvars + v];
      up2RF = uC[qp2R*nvars + v];
    }

    double w1, w2, w3;
    weno_weights_5pt(fm3LF, fm2LF, fm1LF, fp1LF, fp2LF, c1, c2, c3, eps, weight_type, &w1, &w2, &w3);
    w1LF[p*nvars + v] = w1; w2LF[p*nvars + v] = w2; w3LF[p*nvars + v] = w3;
    weno_weights_5pt(fm3RF, fm2RF, fm1RF, fp1RF, fp2RF, c1, c2, c3, eps, weight_type, &w1, &w2, &w3);
    w1RF[p*nvars + v] = w1; w2RF[p*nvars + v] = w2; w3RF[p*nvars + v] = w3;
    weno_weights_5pt(um3LF, um2LF, um1LF, up1LF, up2LF, c1, c2, c3, eps, weight_type, &w1, &w2, &w3);
    w1LU[p*nvars + v] = w1; w2LU[p*nvars + v] = w2; w3LU[p*nvars + v] = w3;
    weno_weights_5pt(um3RF, um2RF, um1RF, up1RF, up2RF, c1, c2, c3, eps, weight_type, &w1, &w2, &w3);
    w1RU[p*nvars + v] = w1; w2RU[p*nvars + v] = w2; w3RU[p*nvars + v] = w3;
  }
}

extern "C" void gpu_launch_weno5_weights(
  const double *fC, const double *uC,
  double *w1, double *w2, double *w3,
  int weno_size_total,
  int ndims, int nvars, const int *dim, const int *stride_with_ghosts,
  int ghosts, int dir, int ip_dir, int iproc_dir,
  int is_crweno,
  int weight_type,
  double eps,
  int blockSize
)
{
  if (!fC || !uC || !w1 || !w2 || !w3 || !dim || !stride_with_ghosts) return;
  if (ndims < 1 || ndims > 3) return;
  if (dir < 0 || dir >= ndims) return;
  if (blockSize <= 0) blockSize = 256;

  const int dim0 = dim[0];
  const int dim1 = (ndims > 1) ? dim[1] : 1;
  const int dim2 = (ndims > 2) ? dim[2] : 1;
  const int stride_dir = stride_with_ghosts[dir];

  const int bi0 = (dir == 0 ? (dim0 + 1) : dim0);
  const int bi1 = (ndims > 1 ? (dir == 1 ? (dim1 + 1) : dim1) : 1);
  const int bi2 = (ndims > 2 ? (dir == 2 ? (dim2 + 1) : dim2) : 1);
  const int ninterfaces = bi0 * bi1 * bi2;

  GPULaunchConfig cfg = GPUConfigureLaunch((size_t)ninterfaces, blockSize);
  gpu_weno5_weights_kernel<<<cfg.gridSize, cfg.blockSize>>>(
    fC, uC, w1, w2, w3, weno_size_total, ndims, nvars,
    dim0, dim1, dim2,
    ghosts, dir, stride_dir, ip_dir, iproc_dir,
    is_crweno, weight_type, eps
  );
}

extern "C" void gpu_launch_weno5_weights_char(
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
)
{
  if (!fC || !uC || !w1 || !w2 || !w3 || !dim || !stride_with_ghosts) return;
  if (ndims < 1 || ndims > 3) return;
  if (dir < 0 || dir >= ndims) return;
  if (blockSize <= 0) blockSize = 256;

  const int dim0 = dim[0];
  const int dim1 = (ndims > 1) ? dim[1] : 1;
  const int dim2 = (ndims > 2) ? dim[2] : 1;
  const int stride_dir = stride_with_ghosts[dir];

  const int bi0 = (dir == 0 ? (dim0 + 1) : dim0);
  const int bi1 = (ndims > 1 ? (dir == 1 ? (dim1 + 1) : dim1) : 1);
  const int bi2 = (ndims > 2 ? (dir == 2 ? (dim2 + 1) : dim2) : 1);
  const int ninterfaces = bi0 * bi1 * bi2;

  GPULaunchConfig cfg = GPUConfigureLaunch((size_t)ninterfaces, blockSize);
  gpu_weno5_weights_char_kernel<<<cfg.gridSize, cfg.blockSize>>>(
    fC, uC, w1, w2, w3, weno_size_total, ndims, nvars,
    dim0, dim1, dim2,
    ghosts, dir, stride_dir, ip_dir, iproc_dir,
    is_crweno, weight_type, eps, gamma
  );
}


