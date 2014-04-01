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

#ifndef EPPMATRIXQ1_SLEP
#define EPPMATRIXQ1_SLEP

#include <shogun/lib/config.h>

#include <shogun/lib/slep/q1/epph.h> /* This is the head file that contains the implementation of the used functions*/

/*
 Lp Norm Regularized Euclidean Projection
 
        min  1/2 ||x- v||_2^2 + rho * ||x||_p
 
 Usage (in Matlab):
 [x, c, iter_step]=epp(v, n, rho, p, c0);

 Usage in C:
 epp(x, c, iter_step, v, n, rho, p, c0);

 The function epp implements the following three functions
 epp1(x, v, n, rho) for p=1
 epp2(x, v, n, rho) for p=2
 eppInf(x, c, iter_step, v,  n, rho, c0) for p=inf
 eppO(x, c, iter_step, v,   n, rho, p) for other p

------------------------------------------------------------

  Here, the input and output are of matrix form. Each row corresponds a group


 Written by Jun Liu, May 18th, 2009
 For any problem, please contact: j.liu@asu.edu
 
 */
void eppMatrix(double *X, double * V, int k, int n, double rho, double p);
#endif   /* ----- #ifndef EPPMATRIXQ1_SLEP  ----- */

