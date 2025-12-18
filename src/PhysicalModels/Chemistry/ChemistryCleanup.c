/*! @file ChemistryCleanup.c
    @author Debojyoti Ghosh
    @brief Contains the function to clean up the chemisty module
*/

#include <stdlib.h>
#include <physicalmodels/chemistry.h>
#include <gpu_runtime.h>
#include <physicalmodels/gpu_chemistry.h>

/*! Function to clean up all physics-related allocations for the photochemical reactions */
  int ChemistryCleanup( void *s /*!< Object of type #Chemistry */ )
{
  Chemistry *param  = (Chemistry*) s;

  // Free GPU memory if allocated
#ifndef GPU_NONE
  if (GPUShouldUse()) {
    GPUChemistryFree(param);
  } else {
    // Only free CPU memory if GPU was not used
    free(param->nv_hnu);
  }
#else
  free(param->nv_hnu);
#endif
  
  free(param->imap);

  return 0;
}
