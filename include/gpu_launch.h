/*! @file gpu_launch.h
    @brief GPU kernel launch function declarations
*/

#ifndef _GPU_LAUNCH_H_
#define _GPU_LAUNCH_H_

#include <gpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel launch wrappers */
void gpu_launch_array_copy(double *dst, const double *src, int n, int blockSize);
void gpu_launch_array_set_value(double *x, double value, int n, int blockSize);
void gpu_launch_array_scale(double *x, double a, int n, int blockSize);
void gpu_launch_array_axpy(const double *x, double a, double *y, int n, int blockSize);
void gpu_launch_array_aypx(const double *x, double a, double *y, int n, int blockSize);
void gpu_launch_array_axby(double *z, double a, const double *x, double b, const double *y, int n, int blockSize);
void gpu_launch_array_scale_copy(const double *x, double a, double *y, int n, int blockSize);
void gpu_launch_array_add(double *z, const double *x, const double *y, int n, int blockSize);
void gpu_launch_array_subtract(double *z, const double *x, const double *y, int n, int blockSize);
void gpu_launch_array_multiply(double *z, const double *x, const double *y, int n, int blockSize);
double gpu_launch_array_max(const double *x, int n, int blockSize);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_LAUNCH_H_ */

