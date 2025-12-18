/*! @file gpu_upwind_function.h
    @brief GPU-enabled upwind function declarations
*/

#ifndef _GPU_UPWIND_FUNCTION_H_
#define _GPU_UPWIND_FUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

/* NavierStokes3D */
int GPUNavierStokes3DUpwindRoe(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUNavierStokes3DUpwindRF(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUNavierStokes3DUpwindLLF(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUNavierStokes3DUpwindRusanov(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);

/* NavierStokes2D - placeholder, falls back to CPU */
int GPUNavierStokes2DUpwindRoe(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUNavierStokes2DUpwindRF(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUNavierStokes2DUpwindLLF(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUNavierStokes2DUpwindRusanov(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);

/* Euler1D - placeholder, falls back to CPU */
int GPUEuler1DUpwindRoe(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUEuler1DUpwindRF(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUEuler1DUpwindLLF(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);
int GPUEuler1DUpwindRusanov(double *fI, double *fL, double *fR, double *uL, double *uR, double *u, int dir, void *s, double t);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_UPWIND_FUNCTION_H_ */
