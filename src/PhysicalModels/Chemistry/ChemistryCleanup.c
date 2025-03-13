/*! @file ChemistryCleanup.c
    @author Debojyoti Ghosh
    @brief Contains the function to clean up the chemisty module
*/

#include <stdlib.h>
#include <physicalmodels/chemistry.h>

/*! Function to clean up all physics-related allocations for the photochemical reactions */
  int ChemistryCleanup( void *s /*!< Object of type #Chemistry */ )
{
  Chemistry *param  = (Chemistry*) s;

  free(param->nv_O2);
  free(param->nv_O3);
  free(param->nv_1D);
  free(param->nv_1Dg);
  free(param->nv_1Sg);
  free(param->nv_CO2);
  free(param->nv_hnu);

  return 0;
}
