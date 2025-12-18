/*! @file gpu_interpolation_function.h
    @brief GPU-enabled InterpolateInterfacesHyp function declarations
*/

#ifndef _GPU_INTERPOLATION_FUNCTION_H_
#define _GPU_INTERPOLATION_FUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

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
);

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
);

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
);

/* GPU version of InterpolateInterfacesHyp for MUSCL2 (characteristic-based; currently NS3D only) */
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
);

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
);

/* GPU version of InterpolateInterfacesHyp for MUSCL3 (characteristic-based; currently NS3D only) */
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
);

/* GPU version of InterpolateInterfacesHyp for first order upwind (component-wise) */
int GPUInterpolateInterfacesHypFirstOrderUpwind(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for first order upwind (characteristic-based) */
int GPUInterpolateInterfacesHypFirstOrderUpwindChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for second order central (component-wise) */
int GPUInterpolateInterfacesHypSecondOrderCentral(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for second order central (characteristic-based) */
int GPUInterpolateInterfacesHypSecondOrderCentralChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for fourth order central (component-wise) */
int GPUInterpolateInterfacesHypFourthOrderCentral(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for fourth order central (characteristic-based) */
int GPUInterpolateInterfacesHypFourthOrderCentralChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for fifth order upwind (component-wise) */
int GPUInterpolateInterfacesHypFifthOrderUpwind(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

/* GPU version of InterpolateInterfacesHyp for fifth order upwind (characteristic-based) */
int GPUInterpolateInterfacesHypFifthOrderUpwindChar(
  double *fI, double *fC, double *u, double *x, int upw, int dir, void *s, void *m, int uflag
);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_INTERPOLATION_FUNCTION_H_ */

