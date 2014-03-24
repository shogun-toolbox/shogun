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

#ifndef  OVERLAPPING_SLEP
#define  OVERLAPPING_SLEP

#include <shogun/lib/config.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


/*
 * -------------------------------------------------------------------
 *                       Function and parameter
 * -------------------------------------------------------------------
 *
 * In this file, we focus solving the following problem
 *
 * 1/2 \|x-v\|^2 + \lambda_1 \|x\|_1 + \lambda_2 \sum w_i \|x_{G_i}\|,
 *
 * where x and v are of dimension p,
 *       w >0, and G_i contains the indices for the i-th group
 *
 * The file is implemented in the following in Matlab:
 *
 * x=overlapping(v, p, g, lambda1, lambda2, w, G, Y, maxIter, flag, tol);
 *
 * x and v are vectors of dimension p
 *
 * g denotes the number of groups
 *
 * lambda1 and labmda2 are non-negative regularization paramter
 *
 * G is a vector containing the indices for the groups
 * G_1, G_2, ..., G_g
 *
 * w is a 3xg matrix
 * w(1,i) contains the starting index of the i-th group in G
 * w(2,i) contains the ending index   of the i-th group in G
 * w(3,i) contains the weight for the i-th group
 *
 * Y is the dual of \|x_{G_i}\|, it is of the same size as G
 *
 * maxIter is the maximal number of iteration
 *
 * flag=0, we apply the pure projected gradient descent 
 *      (forward and backward line search is used)
 *
 * flag=1, we apply the projected gradient descent with restart
 * 
 * in the future, we may apply the accelerated gradient descent 
 *  with adaptive line search (see our KDD'09 paper) with other "flag"
 *
 *
 * Note: 
 * 
 *  1. One should ensure w(2,i)-w(1,i)+1=|G_i|. 
 *      !! The program does not check w(2,i)-w(1,i)+1=|G_i|.!!
 *
 *  2. The index in G and w starts from 0
 *
 * -------------------------------------------------------------------
 *                       History:
 * -------------------------------------------------------------------
 *
 * Composed by Jun Liu on May 17, 2010
 *
 * For any question or suggestion, please email j.liu@asu.edu or
 *                                              jun.liu.80@gmail.com
 *
 */


/*
 * --------------------------------------------------------------------
 *              Identifying some zero Entries
 * --------------------------------------------------------------------
 *
 * lambda1, lambda2 should be non-negative
 *
 * v is the vector of size p to be projected
 *
 *
 * zeroGroupFlag is a vector of size g
 *
 * zeroGroupFlag[i]=0 denotes that the corresponding group is definitely zero
 * zeroGroupFlag[i]=1 denotes that the corresponding group is (possibly) nonzero
 *
 *
 * u is a vector of size p
 *
 *
 * entrySignFlag is a vector of size p
 *
 * entrySignFlag[i]=0 denotes that the corresponding entry is definitely zero
 * entrySignFlag[i]=1 denotes that the corresponding entry is (possibly) positive
 * entrySignFlag[i]=-1 denotes that the corresponding entry is (possibly) negative
 * 
 */
void identifySomeZeroEntries(double * u, int * zeroGroupFlag, int *entrySignFlag,
		int *pp, int *gg,
		double *v, double lambda1, double lambda2, 
		int p, int g, double * w, double *G);

/*
 *
 * function: xFromY
 *
 * compute x=max(u-Y * e, 0);
 *
 * xFromY(x, y, u, Y, p, g, zeroGroupFlag, G, w);
 *
 * y=u-Y * e - max( u - Y * e, 0)
 *
 */
void xFromY(double *x, double *y,
		double *u, double *Y, 
		int p, int g, int *zeroGroupFlag,
		double *G, double *w);

/*
 *
 * function: YFromx
 *
 * compute Y=subgradient(x)
 *
 * YFromx(Y, xnew, Ynew, lambda2, g, zeroGroupFlag, G, w); 
 *
 * The idea is that, if x_{G_i} is nonzero, 
 *           we compute Y^i as x_{G_i}/ \|x_{G_i}\| * lambda2 * w[3*i+2]
 *                   otherwise
 *                      Y^i=Ynew^i
 *
 */
void YFromx(double *Y, 
		double *xnew, double *Ynew,
		double lambda2, int g, int *zeroGroupFlag,
		double *G, double *w);

/*
 * function: dualityGap
 *
 * compute the duality gap for the approximate solution (x, Y)
 *
 * Meanwhile, we compute 
 *       
 *       penalty2=\sum_{i=1}^g w_i \|x_{G_i}\|
 *
 */
void dualityGap(double *gap, double *penalty2,
		double *x, double *Y, int g, int *zeroGroupFlag, 
		double *G, double *w, double lambda2);

/*
 * we solve the proximal opeartor:
 *
 * 1/2 \|x-v\|^2 + \lambda_1 \|x\|_1 + \lambda_2 \sum w_i \|x_{G_i}\|
 *
 * See the description of the variables in the beginning of this file
 *
 * x is the primal variable, each of its entry is non-negative
 *
 * Y is the dual variable, each of its entry should be non-negative
 *
 * flag =0: no restart
 * flag =1; restart
 *
 * tol: the precision parameter
 * 
 * The following code apply the projected gradient descent method 
 *   
 */
void overlapping_gd(double *x, double *gap, double *penalty2,
		double *v, int p, int g, double lambda1, double lambda2,
		double *w, double *G, double *Y, int maxIter, int flag, double tol);

/*
 *
 * do a gradient descent step based (x, Y) to get (xnew, Ynew)
 *
 * (x, Y) is known. Here we do a line search for determining the value of L
 *
 *  gradientDescentStep(double *xnew, double *Ynew, 
 double *LL, double *u,
 double *x, double *Y, int p, int g, int * zeroGroupFlag, 
 double *G, double *w)
 *
 */
void gradientDescentStep(double *xnew, double *Ynew, 
		double *LL, double *u, double *y, int *entrySignFlag, double lambda2,
		double *x, double *Y, int p, int g, int * zeroGroupFlag, 
		double *G, double *w);

/*
 *
 * we use the accelerated gradient descent
 * 
 */
void overlapping_agd(double *x, double *gap, double *penalty2,
		double *v, int p, int g, double lambda1, double lambda2,
		double *w, double *G, double *Y, int maxIter, int flag, double tol);

/*
 * This is main function for the projection
 *
 * It calls overlapping_gd and overlapping_agd based on flag
 *
 * 
 */
void overlapping(double *x, double *gap, double *penalty2,
		double *v, int p, int g, double lambda1, double lambda2,
		double *w, double *G, double *Y, int maxIter, int flag, double tol);

#endif   /* ----- #ifndef OVERLAPPING_SLEP  ----- */
