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

#ifndef  EPPVECTORR_SLEP
#define  EPPVECTORR_SLEP

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/*
   min  1/2 ( ||x- u||_2^2 + ||t-v||_2^2 )
   s.t.  ||x_j||_2 <= t_j
 
 */

void eppVectorR(double *x, double * t, double * u, double * v, double * ind, int n, int k){
    int i, j;
    double temp;

	/* compute the 2 norm of each group
	*/

	for(j=0;j<k;j++){
		temp=0;
		for(i=(int) (ind[j]); i< (int) (ind[j+1]); i++)
			temp+= u[i]* u[i];
        temp=sqrt(temp);
        /*temp contains the 2-norm of of each row of u*/

        if(temp > fabs(v[j])){
           t[j]=(temp + v[j])/2;
           
           for(i=(int) (ind[j]); i< (int) (ind[j+1]); i++)
               x[i]= t[j] / temp * u[i];
        }
        else
           if(temp <= v[j]){
               t[j]=v[j];
                
               for(i=(int) (ind[j]); i< (int) (ind[j+1]); i++)
                   x[i]= u[i];
            }
            else{
                t[j]=0;
                
               for(i=(int) (ind[j]); i< (int) (ind[j+1]); i++)
                   x[i]=0;
            }
              
	}    
}
#endif   /* ----- #ifndef EPPVECTORR_SLEP  ----- */
