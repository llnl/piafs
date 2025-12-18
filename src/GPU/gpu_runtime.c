/*! @file gpu_runtime.c
    @brief GPU runtime control implementation
*/

#include <gpu.h>
#include <gpu_runtime.h>
#include <hypar.h>
#include <stdlib.h>
#include <string.h>

/* Global GPU enable flag (can be set via environment variable or config) */
static int gpu_enabled_global = -1; /* -1 = auto-detect, 0 = disabled, 1 = enabled */
static int gpu_validate_global = -1; /* -1 unknown, 0 off, 1 on */
static int gpu_sync_every_op_global = -1; /* -1 unknown, 0 off, 1 on */

/* Check if GPU should be used */
int GPUShouldUse(void)
{
  /* Check environment variable first */
  const char *env_gpu = getenv("PIAFS_USE_GPU");
  if (env_gpu) {
    if (strcmp(env_gpu, "1") == 0 || strcmp(env_gpu, "yes") == 0 || strcmp(env_gpu, "true") == 0) {
      gpu_enabled_global = 1;
      return GPUIsAvailable();
    } else if (strcmp(env_gpu, "0") == 0 || strcmp(env_gpu, "no") == 0 || strcmp(env_gpu, "false") == 0) {
      gpu_enabled_global = 0;
      return 0;
    }
  }

  /* Auto-detect if not explicitly set - default to enabled if GPU is available */
  if (gpu_enabled_global == -1) {
    gpu_enabled_global = GPUIsAvailable() ? 1 : 0;
  }

  return (gpu_enabled_global == 1) && GPUIsAvailable();
}

/* Expensive debug validation (host copies + scans). Default: off. */
int GPUShouldValidate(void)
{
  if (gpu_validate_global != -1) return gpu_validate_global;
  const char *env = getenv("PIAFS_GPU_VALIDATE");
  if (!env) { gpu_validate_global = 0; return 0; }
  if (strcmp(env, "1") == 0 || strcmp(env, "yes") == 0 || strcmp(env, "true") == 0) {
    gpu_validate_global = 1;
    return 1;
  }
  gpu_validate_global = 0;
  return 0;
}

/* Force sync after small GPU ops (helps debugging, hurts performance). Default: off. */
int GPUShouldSyncEveryOp(void)
{
  if (gpu_sync_every_op_global != -1) return gpu_sync_every_op_global;
  const char *env = getenv("PIAFS_GPU_SYNC_EVERY_OP");
  if (!env) { gpu_sync_every_op_global = 0; return 0; }
  if (strcmp(env, "1") == 0 || strcmp(env, "yes") == 0 || strcmp(env, "true") == 0) {
    gpu_sync_every_op_global = 1;
    return 1;
  }
  gpu_sync_every_op_global = 0;
  return 0;
}

/* Enable GPU for a solver */
void GPUEnable(void *solver)
{
  /* Use global flag - HyPar doesn't have use_gpu field */
  gpu_enabled_global = 1;
  (void) solver; /* unused */
}

/* Disable GPU for a solver */
void GPUDisable(void *solver)
{
  /* Use global flag - HyPar doesn't have use_gpu field */
  gpu_enabled_global = 0;
  (void) solver; /* unused */
}

/* Get GPU status */
int GPUIsEnabled(void *solver)
{
  (void) solver; /* unused */
  return GPUShouldUse();
}

/* Get number of GPU devices */
int GPUGetDeviceCount(void)
{
#ifdef GPU_CUDA
  int count = 0;
  /* Initialize CUDA runtime if needed - cudaGetDeviceCount will do this automatically */
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    /* If error is cudaErrorNoDevice, that's okay - just means no GPUs */
    if (err == cudaErrorNoDevice) {
      return 0;
    }
    /* For other errors, print warning but still return 0 */
    fprintf(stderr, "Warning: cudaGetDeviceCount() failed with error: %s (code: %d)\n", 
            cudaGetErrorString(err), (int)err);
    return 0;
  }
  return count;
#elif defined(GPU_HIP)
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    /* If error is hipErrorNoDevice, that's okay - just means no GPUs */
    if (err == hipErrorNoDevice) {
      return 0;
    }
    /* For other errors, print warning but still return 0 */
    fprintf(stderr, "Warning: hipGetDeviceCount() failed with error: %s (code: %d)\n", 
            hipGetErrorString(err), (int)err);
    return 0;
  }
  return count;
#else
  return 0;
#endif
}

/* Select GPU device */
int GPUSelectDevice(int device_id)
{
#ifdef GPU_CUDA
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: cudaSetDevice(%d) failed: %s\n", 
            device_id, cudaGetErrorString(err));
    return 1;
  }
  return 0;
#elif defined(GPU_HIP)
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    fprintf(stderr, "Error: hipSetDevice(%d) failed: %s\n", 
            device_id, hipGetErrorString(err));
    return 1;
  }
  return 0;
#else
  (void) device_id;
  return 0;
#endif
}

/* Initialize GPU for MPI: select device based on local rank  
   Returns the number of GPU devices available */
int GPUInitializeMPI(int global_rank, int local_rank, int local_size)
{
  int num_devices = GPUGetDeviceCount();
  
  /* Check for CUDA_VISIBLE_DEVICES which may limit visible GPUs */
  const char *cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
  const char *rocr_visible = getenv("ROCR_VISIBLE_DEVICES");
  const char *hip_visible = getenv("HIP_VISIBLE_DEVICES");
  
  if (num_devices <= 0) {
    if (global_rank == 0) {
      fprintf(stderr, "Warning: GPU enabled but no GPU devices found!\n");
      if (cuda_visible) fprintf(stderr, "  CUDA_VISIBLE_DEVICES=%s\n", cuda_visible);
      if (rocr_visible) fprintf(stderr, "  ROCR_VISIBLE_DEVICES=%s\n", rocr_visible);
      if (hip_visible) fprintf(stderr, "  HIP_VISIBLE_DEVICES=%s\n", hip_visible);
    }
    return num_devices; /* Return 0 if no GPUs available */
  }
  
  /* Warn if more local ranks than GPUs (GPU oversubscription) */
  if (local_size > num_devices && local_rank == 0) {
    /* Check if this is GPU isolation (scheduler assigned one GPU per task) */
    int gpu_isolation = (num_devices == 1) && (cuda_visible || rocr_visible || hip_visible);
    
    if (gpu_isolation) {
      /* GPU isolation detected - each rank sees only its assigned GPU as "GPU 0" */
      if (global_rank == 0) {
        fprintf(stderr, "Info: GPU isolation detected (%d ranks, each sees 1 GPU).\n", local_size);
        fprintf(stderr, "      This is expected with --gpus-per-task=1.\n");
        fprintf(stderr, "      Each rank uses a different physical GPU.\n");
      }
    } else {
      /* True GPU oversubscription - multiple ranks will share GPUs */
      fprintf(stderr, "Warning: %d MPI ranks on this node but only %d GPU(s) visible.\n",
              local_size, num_devices);
      fprintf(stderr, "         Multiple ranks will share GPUs (may reduce performance).\n");
      if (cuda_visible) {
        fprintf(stderr, "         Note: CUDA_VISIBLE_DEVICES=%s\n", cuda_visible);
      }
      if (rocr_visible) {
        fprintf(stderr, "         Note: ROCR_VISIBLE_DEVICES=%s\n", rocr_visible);
      }
    }
  }
  
  /* Assign GPU based on local rank (round-robin if more ranks than GPUs) */
  int device_id = local_rank % num_devices;
  
  if (GPUSelectDevice(device_id)) {
    return -1; /* Return -1 on error */
  }
  
  /* Verbose output controlled by environment variable */
  const char *verbose = getenv("PIAFS_GPU_VERBOSE");
  if (verbose && (strcmp(verbose, "1") == 0 || strcmp(verbose, "yes") == 0)) {
    fprintf(stdout, "  Rank %d (local %d/%d): GPU %d of %d", 
            global_rank, local_rank, local_size, device_id, num_devices);
    if (cuda_visible) fprintf(stdout, " [CUDA_VISIBLE_DEVICES=%s]", cuda_visible);
    fprintf(stdout, "\n");
    fflush(stdout);
  }
  
  return num_devices; /* Return number of devices */
}

