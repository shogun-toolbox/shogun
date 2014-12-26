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

#include <shogun/lib/slep/SpInvCoVa/invCov.h>
#include <shogun/mathematics/Math.h>

#include <stdlib.h>
#include <time.h>

void m_Ax(double *Ax, double  *A, double *x, int n, int ith)
{
	int i, j;
	double t;
	for(i=0;i<n;i++){
		if (i==ith)
			continue;
		t=0;
		for(j=0;j<n;j++){
			if (j==ith)
				continue;
			t+=A[i*n+j]* x[j];
			Ax[i]=t;
		}
	}
}


int lassoCD(double *Theta, double *W, double *S, double lambda, int n, int ith, int flag, int maxIter, double fGap, double xGap)
{
	int iter_step, i,j;
	double * Ax, * x;
	double u, v, s_v, t=0, x_new;
	double fun_new,fun_old=-100;
	double x_change;

	Ax=         (double *)malloc(sizeof(double)*n);
	if (Ax==NULL)
	{
		printf("\n Memory allocation failure!");
		return (-1);
	}

	x=          (double *)malloc(sizeof(double)*n);
	if (x==NULL)
	{
		printf("\n Memory allocation failure!");
		free(Ax);
		return (-1);
	}

	/* give x an intialized value, from previously Theta*/
	for(i=0;i<n;i++){
		if (i==ith)
			continue;
		x[i]=Theta[i*n+ith];
	}

	/* Ax contains the derivative*/
	m_Ax(Ax, W, x, n, ith);	

	for (iter_step=0;iter_step<maxIter; iter_step++){

		/*printf("\n Iter: %d",iter_step);*/

		x_change=0;

		for (i=0;i<n;i++){
			if(i==ith)
				continue;

			u=W[i*n + i];

			v=Ax[i]-x[i]*u;

			s_v=S[i*n+ ith]-v;

			if(s_v > lambda)
				x_new= (s_v-lambda) / u;
			else{
				if(s_v < -lambda)
					x_new= (s_v + lambda) / u;
				else
					x_new=0;
			}
			if (x[i]!=x_new){
				for(j=0;j<n;j++){
					if (j==ith)
						continue;
					Ax[j]+= W[j*n+ i]*(x_new - x[i]);
				}
				x_change+=fabs(x[i]-x_new);

				x[i]=x_new;
			}
		}

		fun_new=0;
		t=0;
		for(i=0;i<n;i++){
			if (i==ith)
				continue;
			t+= Ax[i]*x[i] ;
			fun_new+=- S[i*n+ith]*x[i]+ lambda* fabs(x[i]);
		}
		fun_new+=0.5*t;


		/*
		   the Lasso terminates either
		   the change of the function value is less than fGap
		   or the change of the solution in terms of L1 norm is less than xGap
		   or the maximal iteration maxIter has achieved
		   */
		if ( (fabs(fun_new-fun_old) <=fGap) || x_change <=xGap){
			/*printf("\n %d, Fun value: %2.5f",iter_step, fun_new);
			  printf("\n The objective gap between adjacent solutions is less than %e",1e-6);
			  */
			break;
		}
		else{
			/*
			   if(iter_step%10 ==0)
			   printf("\n %d, Fun value: %2.5f",iter_step, fun_new);
			   */
			fun_old=fun_new;
		}
	}

	/*printf("\n %d, Fun value: %2.5f",iter_step, fun_new);*/

	if (flag){        
		t=1/(W[ith*n+ith]-t);
		Theta[ith*n + ith]=t;

		for(i=0;i<n;i++){
			if (i==ith)
				continue;
			W[i*n+ ith]=W[ith*n +i]=Ax[i];
			Theta[i*n+ ith]=Theta[ith*n +i]=-x[i]*t;
		}
	}
	else{
		for(i=0;i<n;i++){
			if (i==ith)
				continue;
			W[i*n+ ith]=W[ith*n +i]=Ax[i];
			Theta[i*n+ ith]=Theta[ith*n +i]=x[i];
		}
	}


	free(Ax); free(x);

	return(iter_step);
}


void invCov(double *Theta, double *W, double *S, double lambda, double sum_S, int n,  
		int LassoMaxIter, double fGap, double xGap, /*for the Lasso (inner iteration)*/
		int maxIter, double xtol)  /*for the outer iteration*/
{
	int iter_step, i,j, ith;
	double * W_old;
	double gap;
	int flag=0;

	W_old=  (double *)malloc(sizeof(double)*n*n);


	if ( W_old==NULL ){
		printf("\n Memory allocation failure!");
		exit (-1);
	}

	for(i=0;i<n;i++)
		for(j=0;j<n;j++){
			if (i==j)
				W_old[i*n+j]=W[i*n+j]=S[i*n+j]+lambda;
			else
				W_old[i*n+j]=W[i*n+j]=S[i*n+j];

			Theta[i*n+j]=0;
		}

	for (iter_step=0;iter_step<=maxIter; iter_step++){
		for(ith=0;ith<n;ith++)	
			lassoCD(Theta, W, S, lambda, n, ith, flag, LassoMaxIter,fGap, xGap);

		if (flag)
			break;

		gap=0;
		for(i=0;i<n;i++)
			for(j=0;j<n;j++){
				gap+=fabs(W[i*n+j]-W_old[i*n+j]);
				W_old[i*n+j]=W[i*n+j];
			}

		/* printf("\n Outer Loop: %d, gap %e\n",iter_step,gap); */


		if ( (gap <= xtol) || (iter_step==maxIter-1) ){
			flag=1;
		}
		/*
		   The outer loop terminates either the difference between ajacent solution in terms of L1 norm is less than xtol, 
		   or the maximal iterations has achieved
		   */
	}

	free(W_old);

	/*return (iter_step);*/
}

