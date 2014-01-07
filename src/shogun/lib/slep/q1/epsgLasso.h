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

#ifndef  EPSGLASSO_SLEP
#define  EPSGLASSO_SLEP

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <lib/slep/q1/epph.h> /* This is the head file that contains the implementation of the used functions*/


/*
 Projection for sgLasso

  min  1/2 \|X - V\|_F^2 + \lambda_1 \|X\|_1 + \lambda_2 \|X\|_{p,1}

 Written by Jun Liu, January 15, 2010
 For any problem, please contact: j.liu@asu.edu
 
 */

void epsgLasso(double *X, double * normx, double * V, int k, int n, double lambda1, double lambda2, int pflag){
    int i, j, *iter_step, nn=n*k, m;
    double *v, *x;
    double normValue,c0=0, c;
    
    v=(double *)malloc(sizeof(double)*n);
    x=(double *)malloc(sizeof(double)*n);
    iter_step=(int *)malloc(sizeof(int)*2);
    
	/*
	initialize normx
	*/
	normx[0]=normx[1]=0;


    /*
     X and V are k x n matrices in matlab, stored in column priority manner
     x corresponds a row of X
	 
	 pflag=2:   p=2
	 pflag=0:   p=inf
     */
    
	/*
	soft thresholding 
	by lambda1

    the results are stored in X
	*/
	for (i=0;i<nn;i++){
		if (V[i]< -lambda1)
			X[i]=V[i] + lambda1;
		else
			if (V[i]> lambda1)
				X[i]=V[i] - lambda1;
			else
				X[i]=0;
	}
	
	/*
	Shrinkage or Truncating
	by lambda2
	*/
	if (pflag==2){
		for(i=0; i<k; i++){

			/*
			process the i-th row, and store it in v
			*/
			normValue=0;

			m=n%5;
			for(j=0;j<m;j++){
				v[j]=X[i + j*k];
			}
			for(j=m;j<n;j+=5){
				v[j  ]=X[i + j*k];
				v[j+1]=X[i + (j+1)*k ];
				v[j+2]=X[i + (j+2)*k];
				v[j+3]=X[i + (j+3)*k];
				v[j+4]=X[i + (j+4)*k];
			}
						
			m=n%5;
			for(j=0;j<m;j++){
				normValue+=v[j]*v[j];
			}
			for(j=m;j<n;j+=5){
				normValue+=v[j]*v[j]+
					       v[j+1]*v[j+1]+
						   v[j+2]*v[j+2]+
						   v[j+3]*v[j+3]+
						   v[j+4]*v[j+4];
			}

			/*
			for(j=0; j<n; j++){
				v[j]=X[i + j*k];

				normValue+=v[j]*v[j];
			}
			*/

			normValue=sqrt(normValue);

			if (normValue<= lambda2){
				for(j=0; j<n; j++)
					X[i + j*k]=0;

				/*normx needs not to be updated*/
			}
			else{

				normx[1]+=normValue-lambda2;
				/*update normx[1]*/

				normValue=(normValue-lambda2)/normValue;

				m=n%5;
				for(j=0;j<m;j++){
					X[i + j*k]*=normValue;
					normx[0]+=fabs(X[i + j*k]);
				}
				for(j=m; j<n;j+=5){
					X[i + j*k]*=normValue;
					X[i + (j+1)*k]*=normValue;
					X[i + (j+2)*k]*=normValue;
					X[i + (j+3)*k]*=normValue;
					X[i + (j+4)*k]*=normValue;

					normx[0]+=fabs(X[i + j*k])+
						      fabs(X[i + (j+1)*k])+
							  fabs(X[i + (j+2)*k])+
							  fabs(X[i + (j+3)*k])+
							  fabs(X[i + (j+4)*k]);
				}

				/*
				for(j=0; j<n; j++)
					X[i + j*k]*=normValue;
				*/
			}
		}
	}
	else{
		for(i=0; i<k; i++){
			
		    /*
			process the i-th row, and store it in v
			*/			
			normValue=0;
			for(j=0; j<n; j++){
				v[j]=X[i + j*k];

				normValue+=fabs(v[j]);
			}

			if (normValue<= lambda2){
				for(j=0; j<n; j++)
					X[i + j*k]=0;
			}
			else{
				eplb(x, &c, iter_step, v, n, lambda2, c0);

				for(j=0; j<n; j++){
					if (X[i + j*k] > c)
						X[i + j*k]=c;
					else
						if (X[i + j*k]<-c)
							X[i + j*k]=-c;
				}
			}
		}
	}

    
    free(v);
    free(x);
    free(iter_step);    
}
#endif   /* ----- #ifndef EPSGLASSO_SLEP  ----- */

