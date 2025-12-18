/*! @file gpu_first_derivative.h
    @brief GPU-enabled first derivative function declarations
*/

#ifndef _GPU_FIRST_DERIVATIVE_H_
#define _GPU_FIRST_DERIVATIVE_H_

/* GPU-enabled first derivative functions */
int GPUFirstDerivativeSecondOrderCentral(double *Df, double *f, int dir, int bias, void *s, void *m);
int GPUFirstDerivativeFourthOrderCentral(double *Df, double *f, int dir, int bias, void *s, void *m);
int GPUFirstDerivativeFirstOrder(double *Df, double *f, int dir, int bias, void *s, void *m);

#endif /* _GPU_FIRST_DERIVATIVE_H_ */

