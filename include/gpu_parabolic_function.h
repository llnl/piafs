/*! @file gpu_parabolic_function.h
    @brief GPU-enabled parabolic function declarations
*/

#ifndef _GPU_PARABOLIC_FUNCTION_H_
#define _GPU_PARABOLIC_FUNCTION_H_

/* GPU-enabled parabolic functions */
int GPUNavierStokes3DParabolicFunction(double *par, double *u, void *s, void *m, double t);
int GPUNavierStokes2DParabolicFunction(double *par, double *u, void *s, void *m, double t);

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif /* _GPU_PARABOLIC_FUNCTION_H_ */

