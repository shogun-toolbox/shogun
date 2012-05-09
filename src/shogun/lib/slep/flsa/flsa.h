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

#ifndef  FLSA_SLEP
#define  FLSA_SLEP

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <shogun/lib/slep/flsa/sfa.h>


/*

   Files contained in this header file sfa.h:

   1. Algorithms for solving the linear system A A^T z0 = Av (see the description of A from the following context)

   void Thomas(double *zMax, double *z0, 
   double * Av, int nn)

   void Rose(double *zMax, double *z0, 
   double * Av, int nn)

   int supportSet(double *x, double *v, double *z, 
   double *g, int * S, double lambda, int nn)

   void dualityGap(double *gap, double *z, 
   double *g, double *s, double *Av, 
   double lambda, int nn)

   void dualityGap2(double *gap, double *z, 
   double *g, double *s, double *Av, 
   double lambda, int nn)


   2. The Subgraident Finding Algorithm (SFA) for solving problem (4) (refer to the description of the problem for detail) 

   int sfa(double *x,     double *gap,
   double *z,     double *z0,   double * v,   double * Av, 
   double lambda, int nn,       int maxStep,
   double *s,     double *g,
   double tol,    int tau,       int flag)

   int sfa_special(double *x,     double *gap,
   double *z,     double * v,   double * Av, 
   double lambda, int nn,       int maxStep,
   double *s,     double *g,
   double tol,    int tau)

   int sfa_one(double *x,     double *gap,
   double *z,     double * v,   double * Av, 
   double lambda, int nn,       int maxStep,
   double *s,     double *g,
   double tol,    int tau)


*/

/*

   In this file, we solve the Fused Lasso Signal Approximator (FLSA) problem:

   min_x  1/2 \|x-v\|^2  + lambda1 * \|x\|_1 + lambda2 * \|A x\|_1,      (1)

   It can be shown that, if x* is the solution to

   min_x  1/2 \|x-v\|^2  + lambda2 \|A x\|_1,                            (2)

   then 
   x**= sgn(x*) max(|x*|-lambda_1, 0)                                    (3)

   is the solution to (1).

   By some derivation (see the description in sfa.h), (2) can be solved by

   x*= v - A^T z*,

   where z* is the optimal solution to

   min_z  1/2  z^T A AT z - < z, A v>,
   subject to  \|z\|_{infty} \leq lambda2                             (4)
   */



/*

   In flsa, we solve (1) corresponding to a given (lambda1, lambda2)

   void flsa(double *x, double *z, double *gap,
   double * v, double *z0, 
   double lambda1, double lambda2, int n, 
   int maxStep, double tol, int flag)

   Output parameters:
x:        the solution to problem (1)
z:        the solution to problem (4)
infor:    the information about running the subgradient finding algorithm
infor[0] = gap:         the computed gap (either the duality gap
or the summation of the absolute change of the adjacent solutions)
infor[1] = steps:       the number of iterations
infor[2] = lambad2_max: the maximal value of lambda2_max
infor[3] = numS:        the number of elements in the support set

Input parameters:
v:        the input vector to be projected
z0:       a guess of the solution of z

lambad1:  the regularization parameter
labmda2:  the regularization parameter
n:        the length of v and x

maxStep:  the maximal allowed iteration steps
tol:      the tolerance parameter
tau:      the program sfa is checked every tau iterations for termination
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

Revision made on October 31, 2009.
The input variable z0 is not modified after calling sfa. For this sake, we allocate a new variable zz to replace z0.
*/



void flsa(double *x, double *z, double *infor,
		double * v, double *z0, 
		double lambda1, double lambda2, int n, 
		int maxStep, double tol, int tau, int flag){

	int i, nn=n-1, m;
	double zMax, temp;
	double *Av, *g, *s;
	int iterStep, numS;
	double gap;
	double *zz; /*to replace z0, so that z0 shall not revised after */


	Av=(double *) malloc(sizeof(double)*nn);

	/*
	   Compute Av= A*v                  (n=4, nn=3)
	   A= [ -1  1  0  0;
	   0  -1 1  0;
	   0  0  -1 1]
	   */

	for (i=0;i<nn; i++)
		Av[i]=v[i+1]-v[i];

	/*
	   Sovlve the linear system via Thomas's algorithm or Rose's algorithm
	   B * z0 = Av
	   */

	Thomas(&zMax, z, Av, nn);

	/*
	   Rose(&zMax, z, Av, nn);
	   */


	/*
	   printf("\n zMax=%2.5f\n",zMax);
	   */


	/*
	   We consider two cases:
	   1) lambda2 >= zMax, which leads to a solution with same entry values
	   2) lambda2 < zMax, which needs to first run sfa, and then perform soft thresholding
	   */


	/*
	   First case: lambda2 >= zMax
	   */
	if (lambda2 >= zMax){

		temp=0;
		m=n%5;
		if (m!=0){
			for (i=0;i<m;i++)
				temp+=v[i];
		}		
		for (i=m;i<n;i+=5){
			temp += v[i] + v[i+1] + v[i+2] + v[i+3] + v[i+4];
		}
		temp/=n; 
		/* temp is the mean value of v*/


		/*
		   soft thresholding by lambda1
		   */
		if (temp> lambda1)
			temp= temp-lambda1;
		else
			if (temp < -lambda1)
				temp= temp+lambda1;
			else
				temp=0;

		m=n%7;
		if (m!=0){
			for (i=0;i<m;i++)
				x[i]=temp;
		}
		for (i=m;i<n;i+=7){
			x[i]   =temp;
			x[i+1] =temp;
			x[i+2] =temp;
			x[i+3] =temp;
			x[i+4] =temp;
			x[i+5] =temp;
			x[i+6] =temp;
		}

		gap=0;

		free(Av);

		infor[0]= gap;
		infor[1]= 0;
		infor[2]=zMax;
		infor[3]=0;

		return;
	}


	/*
	   Second case: lambda2 < zMax

	   We need to call sfa for computing x, and then do soft thresholding

	   Before calling sfa, we need to allocate memory for g and s, 
	   and initialize z and z0.
	   */


	/*
	   Allocate memory for g and s
	   */

	g    =(double *) malloc(sizeof(double)*nn),
		 s    =(double *) malloc(sizeof(double)*nn);



	m=flag /10;
	/* 

	   If m=0, then this shows that, z0 is a "good" starting point. (m=1-6)

	   Otherwise (m=11-16), we shall set z as either the solution to the linear system.
	   or the zero point

*/
	if (m==0){
		for (i=0;i<nn;i++){
			if (z0[i] > lambda2)
				z[i]=lambda2;
			else
				if (z0[i]<-lambda2)
					z[i]=-lambda2;
				else
					z[i]=z0[i];	
		}
	}
	else{
		if (lambda2 >= 0.5 * zMax){
			for (i=0;i<nn;i++){
				if (z[i] > lambda2)
					z[i]=lambda2;
				else
					if (z[i]<-lambda2)
						z[i]=-lambda2;
			}
		}
		else{
			for (i=0;i<nn;i++)
				z[i]=0;

		}
	}

	flag=flag %10;  /*
					   flag is now in [1:6]

					   for sfa, i.e., flag in [1:4], we need initialize z0 with zero
					   */

	if (flag>=1 && flag<=4){
		zz    =(double *) malloc(sizeof(double)*nn);

		for (i=0;i<nn;i++)
			zz[i]=0;
	}

	/*
	   call sfa, sfa_one, or sfa_special to compute z, for finding the subgradient
	   and x
	   */

	if (flag==6)
		iterStep=sfa_one(x, &gap, &numS,
				z,  v,   Av, 
				lambda2, nn,  maxStep,
				s, g,
				tol, tau);
	else
		if (flag==5)
			iterStep=sfa_special(x, &gap, &numS,
					z,  v,   Av, 
					lambda2, nn,  maxStep,
					s, g,
					tol, tau);
		else{
			iterStep=sfa(x, &gap, &numS,
					z, zz,   v,  Av, 
					lambda2, nn, maxStep,
					s,  g,
					tol,tau, flag);

			free (zz);
			/*free the variable zz*/
		}


	/*
	   soft thresholding by lambda1
	   */

	for(i=0;i<n;i++)
		if (x[i] > lambda1)
			x[i]-=lambda1;
		else
			if (x[i]<-lambda1)
				x[i]+=lambda1;
			else
				x[i]=0;


	free(Av);
	free(g);
	free(s);

	infor[0]=gap;
	infor[1]=iterStep;
	infor[2]=zMax;
	infor[3]=numS;
}
#endif   /* ----- #ifndef FLSA_SLEP  ----- */

