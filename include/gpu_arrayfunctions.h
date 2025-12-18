/*! @file gpu_arrayfunctions.h
    @brief GPU-enabled array function wrappers
*/

#ifndef _GPU_ARRAYFUNCTIONS_H_
#define _GPU_ARRAYFUNCTIONS_H_

#include <gpu.h>
#include <arrayfunctions.h>

/* GPU-enabled array operations - these work on device pointers */
void GPUArraySetValue(double *x, double value, int n);
void GPUArrayCopy(double *dst, const double *src, int n);
void GPUArrayScale(double *x, double a, int n);
void GPUArrayAXPY(const double *x, double a, double *y, int n);
void GPUArrayAYPX(const double *x, double a, double *y, int n);
void GPUArrayAXBY(double *z, double a, const double *x, double b, const double *y, int n);
void GPUArrayScaleCopy(const double *x, double a, double *y, int n);
void GPUArrayAdd(double *z, const double *x, const double *y, int n);
void GPUArraySubtract(double *z, const double *x, const double *y, int n);
void GPUArrayMultiply(double *z, const double *x, const double *y, int n);

/* Helper to check if pointer is on GPU */
int GPUIsDevicePtr(const void *ptr);

#endif /* _GPU_ARRAYFUNCTIONS_H_ */

