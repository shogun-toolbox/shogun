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

#ifndef  TESLA_PROJ_SLEP
#define  TESLA_PROJ_SLEP

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <lib/slep/flsa/flsa.h>


/*

   Functions contained in "flsa.h"

   void flsa(double *x, double *z, double *infor,
   double * v, double *z0, 
   double lambda1, double lambda2, int n, 
   int maxStep, double tol, int tau, int flag)

*/

/*
   In this file, we need to make use of the function flsa for solving the following problem

   min 1/2 \|X - V\|_2^2  + lambda1 * \|X\|_1 + lambda2 \|X A^T\|_1   (1)

   where X and V are of size p x n

   For the definition of A, please refer to "flsa.h" and the included "sfa.h".

   The problem can be decoupled into the following 

   min_x  1/2 \|x-v\|^2  + lambda1 * \|x\|_1 + lambda2 * \|A x\|_1,   (2)

   where x and v correspond to a row of of X and V, respectively.

   The problem (2) is essentially the flsa problem, and can be solved by the function flsa.


   void tesla_proj(double *X, double *Z, double *gap,
   double *V, double *Z0,
   double lambda1, double lambda2, int p, int n,
   int maxStep, double tol, int flag)

   Output parameters:
X:          the solution (of size p x n)
Z:          the auxiliary variable (result by subgradient finding),
Z can be used as a warm start for the next "tesla_proj", 
size: p x (n-1)
gap:        the gap for each decoupled flsa problem (of size p x 1)

Input parameters:
V:          the one to be projected
Z0:         the starting point (see flag for whether it is used as the starting point)
size: p x (n-1)

lambda1:    the regularization parameter
lambda2:    the regularization parameter
p:          the number of rows in X and V
n:          the number of columns in X and V

maxStep:    the maximal allowed iteration steps
tol:    the tolerance parameter
flag:     the flag for initialization and deciding calling sfa
switch ( flag )
1-4, 11-14: sfa

switch ( flag )
case 1, 2, 3, or 4: 
z0 is a "good" starting point 
(such as the warm-start of the previous solution,
or the user want to test the performance of this starting point;
the starting point shall be further projected to the L_{infty} ball,
to make sure that it is feasible)

case 11, 12, 13, or 14: z0 is a "random" guess, and thus not used
(we shall initialize z as follows:
if lambda2 >= 0.5 * lambda_2^max, we initialize the solution of the linear system;
if lambda2 <  0.5 * lambda_2^max, we initialize with zero
this solution is projected to the L_{infty} ball)

switch( flag )
5, 15: sfa_special

switch( flag )
5:  z0 is a good starting point
15: z0 is a bad starting point, use the solution of the linear system


switch( flag )
6, 16: sfa_one

switch( flag )
	6:  z0 is a good starting point
	16: z0 is a bad starting point, use the solution of the linear system

	*/

	void tesla_proj(double *X, double *Z, double *gap,
			double *V, double *Z0,
			double lambda1, double lambda2, int p, int n,
			int maxStep, double tol, int tau, int flag){
		/*
		   We assume that X and V are of size p x n
		   */

		int i, j;
		int nn=n-1;
		double
			*x    =(double *) malloc(sizeof(double)*n),	
			*v    =(double *) malloc(sizeof(double)*n),
			*z    =(double *) malloc(sizeof(double)*nn),
			*z0   =(double *) malloc(sizeof(double)*nn),
			*infor=(double *) malloc(sizeof(double)*4);
		//double temp;



		if (n<3){
			printf("\n n should be equal to or larger than 3");
			exit(-1);
		}


		for(i=0;i<p; i++){

			/*
			   copy a row of V to v
			   */
			for (j=0;j<n; j++)
				v[j]=V[j*p + i];

			/*
			   copy a row of Z0 to z0
			   */
			for (j=0;j<nn; j++)
				z0[j]=Z0[j*p + i];

			/*
			   call flsa to compute x
			   */

			flsa(x, z, infor,
					v, z0, 
					lambda1, lambda2, n, 
					maxStep, tol, tau, flag);


			/*
			   store the solution x to X
			   */
			for (j=0;j<n; j++)
				X[j*p + i]=x[j];

			/*
			   store the solution z to Z
			   */
			for (j=0;j<nn; j++)
				Z[j*p + i]=z[j];

			gap[i]=infor[0];
		}


		free(x);
		free(v);
		free(z);
		free(z0);
		free(infor);

	}
#endif   /* ----- #ifndef TESLA_PROJ_SLEP  ----- */

