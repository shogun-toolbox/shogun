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

#ifndef  EPPVECTOR_SLEP
#define  EPPVECTOR_SLEP

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <lib/slep/q1/epph.h> /* This is the head file that contains the implementation of the used functions*/


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

   Here, the input and output are of Vector form.


   Written by Jun Liu, May 18th, 2009
   For any problem, please contact: j.liu@asu.edu

*/

void eppVector(double *x, double * v, int* ind, int k, int n, double * rho, double rho_multiplier, double p){
	int i, *iter_step;
	double c0, c;
	double *px, *pv;

	iter_step=(int *)malloc(sizeof(int)*2);

	c0=0;
	for(i=0; i<k; i++)
	{
		px=x+(int)ind[i];
		pv=v+(int)ind[i];

		epp(px, &c, iter_step, pv, (int)(ind[i+1]-ind[i]), rho[i]*rho_multiplier, p, c0);
	}

	free(iter_step);    
}
#endif   /* ----- #ifndef EPPVECTOR_SLEP  ----- */

