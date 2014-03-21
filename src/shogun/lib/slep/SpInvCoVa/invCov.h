/*   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *   Copyright (C) 2009 - 2012 Jun Liu and Jieping Ye
 */

#ifndef  INVCOV_SLEP
#define  INVCOV_SLEP

#include <shogun/lib/config.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/*
 * A:    n x n
 * x:    n x 1
 * Ax:   n x 1
 *
 * Perform the task of Ax= A* x,
 * where the ith row and column in A, and ith row in x
 * are undefined, so that in Ax, the ith row has no meaning
 */
void m_Ax(double *Ax, double  *A, double *x, int n, int ith);

int lassoCD(double *Theta, double *W, double *S, double lambda, int n,
            int ith, int flag, int maxIter, double fGap, double xGap);

void invCov(double *Theta, double *W, double *S, double lambda,
            double sum_S, int n,
            int LassoMaxIter, double fGap,
            double xGap, /*for the Lasso (inner iteration)*/
            int maxIter, double xtol);  /*for the outer iteration*/

#endif   /* ----- #ifndef INVCOV_SLEP  ----- */
