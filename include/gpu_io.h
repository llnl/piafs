/*! @file gpu_io.h
    @brief GPU-aware I/O function declarations
*/

#ifndef _GPU_IO_H_
#define _GPU_IO_H_

/* GPU-aware I/O functions that copy from/to device only when needed */
int GPUWriteBinary(int ndims, int nvars, int *dim, double *x, double *u, char *f, int *index);
int GPUWriteText(int ndims, int nvars, int *dim, double *x, double *u, char *f, int *index);
int GPUReadBinary(int ndims, int nvars, int *dim, double *x, double *u, char *f);

#endif /* _GPU_IO_H_ */

