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

#ifndef  EP21R_SLEP
#define  EP21R_SLEP

#include <shogun/lib/config.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


/*
   Euclidean Projection onto l_{2,1} Ball

   min  1/2 ||x- u||_2^2 + 1/2 ||t- v||_2^2
   s.t. ||x^j||_{2,1} <= t^j


Usage:
[x, t]=ep21R(u, v, n, k);

*/


void ep21R(double * x, double *t, double * u, double * v, int n, int k)
{
	int i, j, tn=n*k;
	double temp;

	/* compute the 2 norm of each group
	*/

	for(j=0;j<n;j++){
		temp=0;
		for(i=j; i< tn; i+=n)
			temp+= u[i]* u[i];
		temp=sqrt(temp);
		/*temp contains the 2-norm of of each row of u*/

		if(temp > fabs(v[j])){
			t[j]=(temp + v[j])/2;
			for (i=j; i<tn; i+=n)
				x[i]= t[j] / temp * u[i];
		}
		else
			if(temp <= v[j]){
				t[j]=v[j];

				for (i=j; i<tn; i+=n)
					x[i]= u[i];
			}
			else{
				t[j]=0;

				for (i=j; i<tn; i+=n)
					x[i]=0;
			}

	}
}
#endif   /* ----- #ifndef EP21R_SLEP  ----- */

