/*! @file gpu_config.h
    @brief GPU runtime configuration and tuning parameters
*/

#ifndef _GPU_CONFIG_H_
#define _GPU_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Get optimal block size for a given operation type and problem size
   operation types: "memory_bound", "compute_bound", "reduction", "stencil"
*/
int GPUGetBlockSize(const char *operation, int n);

/* Get optimal grid size for a given block size and problem size */
int GPUGetGridSize(int n, int blockSize);

/* Print GPU configuration info (only on rank 0) */
void GPUPrintConfig(int rank);

#ifdef __cplusplus
}
#endif

#endif /* _GPU_CONFIG_H_ */
