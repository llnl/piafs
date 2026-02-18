// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file BCCleanup.c
    @author Debojyoti Ghosh
    @brief Function to clean up boundary-conditions related variables
*/
#include <stdlib.h>
#include <boundaryconditions.h>

/*! Cleans up a boundary object of type #DomainBoundary */
int BCCleanup(  void *b /*!< Boundary object of type #DomainBoundary*/ )
{
  DomainBoundary *boundary = (DomainBoundary*) b;
  free(boundary->xmin);
  free(boundary->xmax);
  free(boundary->is);
  free(boundary->ie);
  if (boundary->DirichletValue) free(boundary->DirichletValue);
  if (boundary->SpongeValue   ) free(boundary->SpongeValue   );
  if (boundary->FlowVelocity  ) free(boundary->FlowVelocity  );
  if (boundary->scalars       ) free(boundary->scalars       );

  return 0;
}
