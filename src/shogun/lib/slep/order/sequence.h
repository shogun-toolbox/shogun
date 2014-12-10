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

#ifndef  SEQUENCE_SLEP
#define  SEQUENCE_SLEP

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <shogun/mathematics/Math.h>


/*
 * In this file, we propose the algorithms for solving the problem:
 *
 * min   1/2 \|x - u\|^2
 * s.t.  x1 \ge x2 \ge x3 \ge ... \ge xn \ge 0
 *
 *
 */

/*
 *
 * x= sequence_bottomup(u,n)
 *
 * we compute using a bottom up order
 * 
 */

void sequence_bottomup(double *x, double *u, int n){
	int i, j;
	int *location=(int *)malloc(sizeof(int)*n);
	int num;

	if(!location){
		printf("\n Allocation of array failure!");
		exit(1);
	}


	/*
	 * compute the maximal mean from the root to the i-th point:
	 *
	 * x[i]: the maximal mean
	 * location[i]: the ending index of the mean
	 *
	 */


	/* process the last element*/
	if (n<1){
		printf("\n n=%d should be an integer over 1!",n);
		exit(1);
	}
	else{
		i=n-1;        
		x[i]=u[i];
		location[i]=i; 
	}

	/*process the remaining elements in a bottom-up recursive manner*/
	for(i=n-2;i>=0;i--){


		if (u[i]>x[i+1]){
			x[i]=u[i];
			location[i]=i;            
		}
		else{
			/*make use of x[i: (n-1)] and location[i: (n-1)] for update*/

			/*merge with the first group*/
			num=location[i+1]-i;
			x[i]=(u[i] + x[i+1]*num)/(num+1);
			location[i]=location[i+1];
			j=location[i+1]+1;

			/*If necessary, we need to further merge with the remainig groups */
			for(;j<n;){
				if(x[i] <= x[j]){

					num=location[j]-j +1;
					x[i]=( x[i]* (j-i) + x[j]* num ) / (location[j] -i +1);
					location[i]=location[j];

					j=location[j]+1;
				}
				else
					break;
			}                
		}
	}

	/*
	   for(i=0;i<30;i++)
	   printf("\n x[%d]=%2.5f, location[%d]=%d",i+1, x[i], i+1, location[i]+1);
	   */

	/*
	 * compute the solution x with the mean and location
	 *
	 */

	for(i=0;i<n;){

		if (x[i]>0){
			for(j=i+1;j<=location[i];j++){
				x[j]=x[i];
			}

			i=location[i]+1;
		}
		else{
			for(j=i;j<n;j++)
				x[j]=0;
			break;
		}
	}

	free(location);
}


/*
 *
 * x= sequence_topdown(u,n)
 *
 * we compute using a top to down order
 * 
 */

void sequence_topdown(double *x, double *u, int n){
	int i, j;
	double sum, max, mean;
	int num;
	int *location=(int *)malloc(sizeof(int)*n);


	if(!location){
		printf("\n Allocation of array failure!");
		exit(1);
	}

	for(i=0;i<n;){

		/*
		 * From each root node i, we compute the maximal mean from.
		 *
		 */

		max=0;
		location[i]=i;

		sum=0;
		num=1;        
		for(j=i;j<n;j++){
			sum+=u[j];
			mean=sum/num;            
			num++;

			/* record the most largest mean and the location*/
			if (mean >= max){                
				max=mean;
				location[i]=j;
			}
		}

		if (max>0){
			x[i]=max; /*record the maximal mean*/
			i=location[i]+1; /*the next i*/
		}
		else{
			x[i]=-1; /* any value less or equal to 0
					  *
					  * This shows that the remaining elements should be zero
					  *
					  */
			break;
		}
	}


	/*
	 * compute the solution x with the mean and location
	 *
	 */

	for(i=0;i<n;){

		if (x[i]>0){
			for(j=i+1;j<=location[i];j++){
				x[j]=x[i];
			}

			i=location[i]+1;
		}
		else{
			for(j=i;j<n;j++)
				x[j]=0;
			break;
		}
	}

	free(location);
}
#endif   /* ----- #ifndef SEQUENCE_SLEP  ----- */

