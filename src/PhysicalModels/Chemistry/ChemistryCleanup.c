// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

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

  free(param->nv_hnu);
  free(param->imap);

  return 0;
}
