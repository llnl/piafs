/*! @file gpu_config.c
    @brief GPU runtime configuration and tuning parameters
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global GPU configuration parameters */
static int gpu_block_size_global = -1; /* -1 = auto-select */
static int gpu_config_initialized = 0;

/* Initialize GPU configuration from environment variables */
static void gpu_config_init(void)
{
  if (gpu_config_initialized) return;

  const char *env_block_size = getenv("PIAFS_GPU_BLOCK_SIZE");
  if (env_block_size) {
    int block_size = atoi(env_block_size);
    if (block_size > 0 && block_size <= 1024 && (block_size % 32 == 0)) {
      gpu_block_size_global = block_size;
    } else {
      fprintf(stderr, "Warning: Invalid PIAFS_GPU_BLOCK_SIZE=%s (must be 32-1024 and multiple of 32)\n",
              env_block_size);
    }
  }

  gpu_config_initialized = 1;
}

/* Get optimal block size for a given operation type and problem size */
int GPUGetBlockSize(const char *operation, int n)
{
  gpu_config_init();

  /* If user specified block size, use it */
  if (gpu_block_size_global > 0) {
    return gpu_block_size_global;
  }

  /* Auto-select based on operation type and problem size */
  if (strcmp(operation, "memory_bound") == 0) {
    /* Memory-bound operations (copy, scale, add) - fewer threads, more blocks */
    return (n < 10000) ? 128 : 256;
  } else if (strcmp(operation, "compute_bound") == 0) {
    /* Compute-bound operations (interpolation, flux) - more threads per block */
    return (n < 10000) ? 256 : 512;
  } else if (strcmp(operation, "reduction") == 0) {
    /* Reduction operations - power of 2 for efficient tree reduction */
    return 256;
  } else if (strcmp(operation, "stencil") == 0) {
    /* Stencil operations - balance between shared memory and occupancy */
    return 256;
  } else {
    /* Default for unknown operations */
    return 256;
  }
}

/* Get optimal grid size for a given block size and problem size */
int GPUGetGridSize(int n, int blockSize)
{
  return (n + blockSize - 1) / blockSize;
}

/* Print GPU configuration info */
void GPUPrintConfig(int rank)
{
  if (rank != 0) return;

  gpu_config_init();

  fprintf(stdout, "GPU Configuration:\n");
  if (gpu_block_size_global > 0) {
    fprintf(stdout, "  Block size: %d (user-specified via PIAFS_GPU_BLOCK_SIZE)\n",
            gpu_block_size_global);
  } else {
    fprintf(stdout, "  Block size: auto-select (memory_bound: 128-256, compute_bound: 256-512)\n");
    fprintf(stdout, "  Set PIAFS_GPU_BLOCK_SIZE to override (32-1024, multiple of 32)\n");
  }
  fflush(stdout);
}
