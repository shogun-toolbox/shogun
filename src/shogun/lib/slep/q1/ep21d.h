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

#ifndef  EP21D_SLEP
#define  EP21D_SLEP

#include <shogun/lib/config.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <shogun/lib/slep/q1/epph.h> /* This is the head file that contains the implementation of the used functions*/

/*
   Euclidean Projection onto l_{2,1} Ball

   min  1/2 ||X- V||_2^2
   s.t. ||X||_{2,1} <= z

   which is converted to the following zero finding problem

   f(lambda)= \sum_i ( max( |v^i|-lambda,0) )-z=0

   v^i denotes the i-th row of V

Usage:
[x, lambda, iter_step]=ep21d(y, n, k, z, lambda0);

*/


void ep21d(double * x, double *root, int * steps, double * v, int n, int k, double z, double lambda0)
{
	int i, j, tn=n*k;
	double *vnorm=(double *)malloc(sizeof(double)*n);
	double *vproj=(double *)malloc(sizeof(double)*n);
	double t;

	/* compute the 2 norm of each group
	*/

	for(j=0;j<n;j++){
		t=0;
		for(i=j; i< tn; i+=n)
			t+= v[i]* v[i];
		vnorm[j]=sqrt(t);
	}



	eplb(vproj, root, steps, vnorm, n, z, lambda0);

	/* compute x
	*/

	if (*root==0){
		for(i=0;i<tn;i++)
			x[i]=v[i];
	}
	else{
		for (j=0;j<n;j++){
			if ( vnorm[j] <= *root){
				for(i=j; i< tn; i+=n)
					x[i]=0;
			}
			else{
				t=1- *root/ vnorm[j];
				for(i=j; i< tn; i+=n)
					x[i]=t* v[i];
			}
		}
	}

	free(vnorm);
	free(vproj);

}
#endif   /* ----- #ifndef EP21D_SLEP  ----- */

