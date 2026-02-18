// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file IncrementFilename.c
    @author Debojyoti Ghosh
    @brief Functions for incrementing filename indices
*/

#include <stdio.h>

/*! Increment a character string representing an integer by 1. For example:
    "00002" -> "00003"; "3421934" -> "3421935"; "999" -> "000". The string
    can be of arbitrary length.
*/
void IncrementFilenameIndex(
                              char *f,  /*!< Character string representing the integer */
                              int len   /*!< Length of the string */
                           )
{
  int i;
  for (i=len-1; i>=0; i--) {
    if (f[i] == '9') {
      f[i] = '0';
      if (!i) fprintf(stderr,"Warning: file increment hit max limit. Resetting to zero.\n");
    } else {
      f[i]++;
      break;
    }
  }
}

/*! Resets the index to "0000..." of a desired length. */
void ResetFilenameIndex(  char *f,  /*!< Character string representing the integer */
                          int len   /*!< Length of the string */ )
{
  if (!f) return;
  int i;
  for (i = 0; i < len; i++) {
    f[i] = '0';
  }
  return;
}
