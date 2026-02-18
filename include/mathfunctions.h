// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2018, Lawrence Livermore National Security, LLC

/*! @file mathfunctions.h
    @brief Contains function definitions for common mathematical functions.
    @author Debojyoti Ghosh
 */

#ifndef _MATHFUNC_H_
#define _MATHFUNC_H_

#include <math_ops.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! Function to calculate the grid points corresponding to a given interval */
void FindInterval(double,double,double*,int,int*,int*);

/*! Fill the ghost cells of a global n-dimensional array */
void fillGhostCells(const int* const, const int, double* const, const int, const int, const int* const);

/*! Function to compute trilinear interpolation coefficients */
void TrilinearInterpCoeffs(double,double,double,double,double,double,double,double,double,double*);

/*! Interpolate an n-dimensional grid variable from one grid to another */
int InterpolateGlobalnDVar( const int* const,
                            double** const,
                            const int* const,
                            double* const,
                            const int,
                            const int,
                            const int,
                            const int* const);

#ifdef __cplusplus
}
#endif

#endif
