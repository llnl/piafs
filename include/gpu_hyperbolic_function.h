/*! @file gpu_hyperbolic_function.h
    @brief GPU-enabled HyperbolicFunction declaration
*/

#ifndef _GPU_HYPERBOLIC_FUNCTION_H_
#define _GPU_HYPERBOLIC_FUNCTION_H_

/* GPU-enabled HyperbolicFunction */
int GPUHyperbolicFunction(
  double *hyp,
  double *u,
  void *s,
  void *m,
  double t,
  int LimFlag,
  int(*FluxFunction)(double*,double*,int,void*,double),
  int(*UpwindFunction)(double*,double*,double*,double*,double*,double*,int,void*,double)
);

#endif /* _GPU_HYPERBOLIC_FUNCTION_H_ */

