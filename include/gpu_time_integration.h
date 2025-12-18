/*! @file gpu_time_integration.h
    @brief GPU-aware time integration function declarations
*/

#ifndef _GPU_TIME_INTEGRATION_H_
#define _GPU_TIME_INTEGRATION_H_

/* GPU-aware time integration functions */
int GPUTimePreStep(void *ts);
int GPUTimeRHSFunctionExplicit(double *rhs, double *u, void *s, void *m, double t);
int GPUInitializeTimeIntegrationArrays(void *ts);

#endif /* _GPU_TIME_INTEGRATION_H_ */

