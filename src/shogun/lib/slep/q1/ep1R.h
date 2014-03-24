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

#ifndef  EP1R_SLEP
#define  EP1R_SLEP


#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


/*
   Euclidean Projection onto l_{2,1} Ball

   min  1/2 ||x- u||_2^2 + 1/2 ||t- v||_2^2
   s.t. |x|<=t


Usage:
[x, t]=ep1R(u, v, n);

*/


void ep1R(double * x, double *t, double * u, double * v, int n)
{
	int j;


	for(j=0;j<n;j++){

		if(fabs(u[j]) > fabs(v[j])){
			t[j]=(fabs(u[j]) + v[j])/2;

			if (u[j] >0)
				x[j]=t[j];
			else
				x[j]=-t[j];
		}
		else
			if(fabs(u[j]) <= v[j]){
				t[j]=v[j];
				x[j]=u[j];
			}
			else{
				t[j]=x[j]=0;
			}

	}
}
#endif   /* ----- #ifndef EP1R_SLEP  ----- */

