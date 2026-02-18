// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file common.h
    @brief Some common functions used here and there
    @author Debojyoti Ghosh
*/

#ifndef _COMMON_H_
#define _COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

/*! Get a string corresponding to an integer, i.e. 41 gives "00041" if
    \a width is 5, or "41" if \a width is 2, or "1" if \a width is 1.
*/
void GetStringFromInteger(int, char*, int);

/*! Take the natural logarithm of each element of the array
*/
void takeLog(double* , int);

/*! Take the exponential of each element of the array
*/
void takeExp(double* , int);

/*! Check for NaN or Inf values in an array and abort if found
*/
void checkNanInf(const double* const array, const int array_size, const char* const location);

#ifdef __cplusplus
}
#endif

#endif
