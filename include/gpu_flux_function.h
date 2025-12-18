/*! @file gpu_flux_function.h
    @brief GPU-enabled flux function declarations
*/

#ifndef _GPU_FLUX_FUNCTION_H_
#define _GPU_FLUX_FUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

int GPUNavierStokes3DFlux(double *f, double *u, int dir, void *s, double t);
int GPUNavierStokes2DFlux(double *f, double *u, int dir, void *s, double t);
int GPUEuler1DFlux(double *f, double *u, int dir, void *s, double t);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_FLUX_FUNCTION_H_ */
