/*! @file gpu_runtime.h
    @brief GPU runtime control functions
*/

#ifndef _GPU_RUNTIME_H_
#define _GPU_RUNTIME_H_

#include <gpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Check if GPU should be used based on runtime flags */
int GPUShouldUse(void);

/* Optional runtime debugging/performance controls */
int GPUShouldValidate(void);     /* expensive host-side validation (NaN/rho checks) */
int GPUShouldSyncEveryOp(void);  /* force device sync after small GPU ops */

/* Enable/disable GPU for a solver */
void GPUEnable(void *solver);
void GPUDisable(void *solver);

/* Get GPU status */
int GPUIsEnabled(void *solver);

/* Get number of GPU devices */
int GPUGetDeviceCount(void);

/* Select GPU device for current MPI rank */
int GPUSelectDevice(int device_id);

/* Initialize GPU for MPI: select device based on local rank */
int GPUInitializeMPI(int global_rank, int local_rank, int local_size);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_RUNTIME_H_ */

