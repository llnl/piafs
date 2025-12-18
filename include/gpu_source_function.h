/*! @file gpu_source_function.h
    @brief GPU-enabled NavierStokes3DSource function declaration
*/

#ifndef _GPU_SOURCE_FUNCTION_H_
#define _GPU_SOURCE_FUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

int GPUNavierStokes3DSource(double *source, double *u, void *s, void *m, double t);
int GPUNavierStokes2DSource(double *source, double *u, void *s, void *m, double t);
int GPUEuler1DSource(double *source, double *u, void *s, void *m, double t);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_SOURCE_FUNCTION_H_ */

