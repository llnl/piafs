// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file WENOCleanup.c
    @brief Cleans up allocations specific to WENO-type methods
    @author Debojyoti Ghosh
*/

#include <stdlib.h>
#include <interpolation.h>
#include <mpivars.h>
#include <hypar.h>
#if defined(GPU_CUDA) || defined(GPU_HIP)
#include <gpu.h>
#endif

/*!
    Cleans up all allocations related to the WENO-type methods.
*/
int WENOCleanup(void *s /*!< WENO object of type #WENOParameters */)
{
  WENOParameters  *weno   = (WENOParameters*) s;

  if (weno->offset) free(weno->offset);
  if (weno->w1) free(weno->w1);
  if (weno->w2) free(weno->w2);
  if (weno->w3) free(weno->w3);
#if defined(GPU_CUDA) || defined(GPU_HIP)
  if (weno->w1_gpu) GPUFree(weno->w1_gpu);
  if (weno->w2_gpu) GPUFree(weno->w2_gpu);
  if (weno->w3_gpu) GPUFree(weno->w3_gpu);
#endif

  return(0);
}
