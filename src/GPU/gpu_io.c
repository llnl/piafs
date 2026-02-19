/*! @file gpu_io.c
    @brief GPU-aware I/O functions that copy from device only when needed
*/

#include <stdio.h>
#include <stdlib.h>
#include <gpu.h>
#include <gpu_runtime.h>
#include <gpu_initialize.h>
#include <arrayfunctions.h>

/* GPU-aware WriteBinary: copies from device only for I/O */
int GPUWriteBinary(
  int ndims,
  int nvars,
  int *dim,
  double *x,      /* may be on GPU */
  double *u,      /* may be on GPU */
  char *f,
  int *index
)
{
  int size, d;
  size_t bytes;
  FILE *out;

  /* Calculate sizes */
  int grid_size = 0;
  for (d = 0; d < ndims; d++) grid_size += dim[d];
  int sol_size = 1;
  for (d = 0; d < ndims; d++) sol_size *= dim[d];
  sol_size *= nvars;

  /* Allocate host buffers */
  double *x_host = NULL;
  double *u_host = NULL;

  x_host = (double*) malloc(grid_size * sizeof(double));
  u_host = (double*) malloc(sol_size * sizeof(double));

  if (!x_host || !u_host) {
    fprintf(stderr, "Error: Failed to allocate host buffers for I/O\n");
    if (x_host) free(x_host);
    if (u_host) free(u_host);
    return 1;
  }

  /* Copy from device to host */
  if (GPUShouldUse()) {
    GPUCopyToHost(x_host, x, grid_size * sizeof(double));
    GPUCopyToHost(u_host, u, sol_size * sizeof(double));
    /* GPUCopyToHost is synchronous - no explicit sync needed */
  } else {
    /* Already on host, just copy */
    for (int i = 0; i < grid_size; i++) x_host[i] = x[i];
    for (int i = 0; i < sol_size; i++) u_host[i] = u[i];
  }

  /* Write to file */
  out = fopen(f, "wb");
  if (!out) {
    fprintf(stderr, "Error: could not open %s for writing.\n", f);
    free(x_host);
    free(u_host);
    return 1;
  }

  bytes = fwrite(&ndims, sizeof(int), 1, out);
  if ((int)bytes != 1) {
    fprintf(stderr, "Error in GPUWriteBinary(): Unable to write ndims.\n");
    fclose(out);
    free(x_host);
    free(u_host);
    return 1;
  }

  bytes = fwrite(&nvars, sizeof(int), 1, out);
  if ((int)bytes != 1) {
    fprintf(stderr, "Error in GPUWriteBinary(): Unable to write nvars.\n");
    fclose(out);
    free(x_host);
    free(u_host);
    return 1;
  }

  bytes = fwrite(dim, sizeof(int), ndims, out);
  if ((int)bytes != ndims) {
    fprintf(stderr, "Error in GPUWriteBinary(): Unable to write dimensions.\n");
    fclose(out);
    free(x_host);
    free(u_host);
    return 1;
  }

  bytes = fwrite(x_host, sizeof(double), grid_size, out);
  if ((int)bytes != grid_size) {
    fprintf(stderr, "Error in GPUWriteBinary(): Unable to write grid.\n");
    fclose(out);
    free(x_host);
    free(u_host);
    return 1;
  }

  bytes = fwrite(u_host, sizeof(double), sol_size, out);
  if ((int)bytes != sol_size) {
    fprintf(stderr, "Error in GPUWriteBinary(): Unable to write solution.\n");
    fclose(out);
    free(x_host);
    free(u_host);
    return 1;
  }

  fclose(out);
  free(x_host);
  free(u_host);
  return 0;
}

/* GPU-aware WriteText: copies from device only for I/O */
int GPUWriteText(
  int ndims,
  int nvars,
  int *dim,
  double *x,      /* may be on GPU */
  double *u,      /* may be on GPU */
  char *f,
  int *index
)
{
  FILE *out;

  /* Calculate sizes */
  int grid_size = 0;
  for (int d = 0; d < ndims; d++) grid_size += dim[d];
  int sol_size = 1;
  for (int d = 0; d < ndims; d++) sol_size *= dim[d];
  sol_size *= nvars;

  /* Allocate host buffers */
  double *x_host = (double*) malloc(grid_size * sizeof(double));
  double *u_host = (double*) malloc(sol_size * sizeof(double));

  if (!x_host || !u_host) {
    fprintf(stderr, "Error: Failed to allocate host buffers for I/O\n");
    if (x_host) free(x_host);
    if (u_host) free(u_host);
    return 1;
  }

  /* Copy from device to host */
  if (GPUShouldUse()) {
    GPUCopyToHost(x_host, x, grid_size * sizeof(double));
    GPUCopyToHost(u_host, u, sol_size * sizeof(double));
    /* GPUCopyToHost is synchronous - no explicit sync needed */
  } else {
    /* Already on host */
    for (int i = 0; i < grid_size; i++) x_host[i] = x[i];
    for (int i = 0; i < sol_size; i++) u_host[i] = u[i];
  }

  /* Write to file */
  out = fopen(f, "w");
  if (!out) {
    fprintf(stderr, "Error: could not open %s for writing.\n", f);
    free(x_host);
    free(u_host);
    return 1;
  }

  int done = 0;
  _ArraySetValue_(index, ndims, 0);
  while (!done) {
    int i, p;
    _ArrayIndex1D_(ndims, dim, index, 0, p);
    for (i = 0; i < ndims; i++) fprintf(out, "%4d ", index[i]);
    for (i = 0; i < ndims; i++) {
      int j, offset = 0;
      for (j = 0; j < i; j++) offset += dim[j];
      fprintf(out, "%+1.16E ", x_host[offset + index[i]]);
    }
    for (i = 0; i < nvars; i++) fprintf(out, "%+1.16E ", u_host[nvars*p + i]);
    fprintf(out, "\n");
    _ArrayIncrementIndex_(ndims, dim, index, done);
  }

  fclose(out);
  free(x_host);
  free(u_host);
  return 0;
}

/* GPU-aware ReadBinary: copies to device after reading */
int GPUReadBinary(
  int ndims,
  int nvars,
  int *dim,
  double *x,      /* will be on GPU if available */
  double *u,      /* will be on GPU if available */
  char *f
)
{
  int size, d;
  size_t bytes;
  FILE *in;

  in = fopen(f, "rb");
  if (!in) {
    fprintf(stderr, "Error: could not open %s for reading.\n", f);
    return 1;
  }

  /* Read header */
  int file_ndims, file_nvars;
  bytes = fread(&file_ndims, sizeof(int), 1, in);
  if ((int)bytes != 1 || file_ndims != ndims) {
    fprintf(stderr, "Error in GPUReadBinary(): ndims mismatch.\n");
    fclose(in);
    return 1;
  }

  bytes = fread(&file_nvars, sizeof(int), 1, in);
  if ((int)bytes != 1 || file_nvars != nvars) {
    fprintf(stderr, "Error in GPUReadBinary(): nvars mismatch.\n");
    fclose(in);
    return 1;
  }

  int file_dim[ndims];
  bytes = fread(file_dim, sizeof(int), ndims, in);
  if ((int)bytes != ndims) {
    fprintf(stderr, "Error in GPUReadBinary(): Unable to read dimensions.\n");
    fclose(in);
    return 1;
  }

  /* Calculate sizes */
  int grid_size = 0;
  for (d = 0; d < ndims; d++) grid_size += dim[d];
  int sol_size = 1;
  for (d = 0; d < ndims; d++) sol_size *= dim[d];
  sol_size *= nvars;

  /* Allocate host buffers for reading */
  double *x_host = (double*) malloc(grid_size * sizeof(double));
  double *u_host = (double*) malloc(sol_size * sizeof(double));

  if (!x_host || !u_host) {
    fprintf(stderr, "Error: Failed to allocate host buffers for I/O\n");
    fclose(in);
    if (x_host) free(x_host);
    if (u_host) free(u_host);
    return 1;
  }

  /* Read from file */
  bytes = fread(x_host, sizeof(double), grid_size, in);
  if ((int)bytes != grid_size) {
    fprintf(stderr, "Error in GPUReadBinary(): Unable to read grid.\n");
    fclose(in);
    free(x_host);
    free(u_host);
    return 1;
  }

  bytes = fread(u_host, sizeof(double), sol_size, in);
  if ((int)bytes != sol_size) {
    fprintf(stderr, "Error in GPUReadBinary(): Unable to read solution.\n");
    fclose(in);
    free(x_host);
    free(u_host);
    return 1;
  }

  fclose(in);

  /* Copy to device if GPU is available */
  if (GPUShouldUse()) {
    GPUCopyToDevice(x, x_host, grid_size * sizeof(double));
    GPUCopyToDevice(u, u_host, sol_size * sizeof(double));
    /* GPUCopyToDevice is synchronous - no explicit sync needed unless debugging */
    if (GPUShouldSyncEveryOp()) GPUSync();
  } else {
    /* Copy to host arrays */
    for (int i = 0; i < grid_size; i++) x[i] = x_host[i];
    for (int i = 0; i < sol_size; i++) u[i] = u_host[i];
  }

  free(x_host);
  free(u_host);
  return 0;
}

