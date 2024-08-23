/*! @file WENOCleanup.c
    @brief Cleans up allocations specific to WENO-type methods
    @author Debojyoti Ghosh
*/

#include <stdlib.h>
#include <interpolation.h>
#include <mpivars.h>
#include <hypar.h>

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

  return(0);
}
