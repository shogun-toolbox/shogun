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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/slep/q1/epph.h>

#define delta 1e-8

#define innerIter 1000
#define outerIter 1000

void eplb(double * x, double *root, int * steps, double * v,int n, double z, double lambda0)
{

	int i, j, flag=0;
	int rho_1, rho_2, rho, rho_T, rho_S;
	int V_i_b, V_i_e, V_i;
	double lambda_1, lambda_2, lambda_T, lambda_S, lambda;
	double s_1, s_2, s, s_T, s_S, v_max, temp;
	double f_lambda_1, f_lambda_2, f_lambda, f_lambda_T, f_lambda_S;
	int iter_step=0;

	/* find the maximal absolute value in v
	 * and copy the (absolute) values from v to x
	 */

	if (z< 0){
		printf("\n z should be nonnegative!");
		return;
	}

	V_i=0;
	if (v[0] !=0){
		rho_1=1;
		s_1=x[V_i]=v_max=fabs(v[0]);
		V_i++;
	}
	else{
		rho_1=0;
		s_1=v_max=0;
	}

	for (i=1;i<n; i++){
		if (v[i]!=0){
			x[V_i]=fabs(v[i]); s_1+= x[V_i]; rho_1++;

			if (x[V_i] > v_max)
				v_max=x[V_i];
			V_i++;
		}
	}

	/* If ||v||_1 <= z, then v is the solution  */
	if (s_1 <= z){
		flag=1;        lambda=0;
		for(i=0;i<n;i++){
			x[i]=v[i];
		}
		*root=lambda;
		*steps=iter_step;
		return;
	}

	lambda_1=0; lambda_2=v_max;
	f_lambda_1=s_1 -z;
	/*f_lambda_1=s_1-rho_1* lambda_1 -z;*/
	rho_2=0; s_2=0; f_lambda_2=-z;
	V_i_b=0; V_i_e=V_i-1;

	lambda=lambda0;
	if ( (lambda<lambda_2) && (lambda> lambda_1) ){
		/*-------------------------------------------------------------------
		  Initialization with the root
		 *-------------------------------------------------------------------
		 */

		i=V_i_b; j=V_i_e; rho=0; s=0;
		while (i <= j){
			while( (i <= V_i_e) && (x[i] <= lambda) ){
				i++;
			}
			while( (j>=V_i_b) && (x[j] > lambda) ){
				s+=x[j];
				j--;
			}
			if (i<j){
				s+=x[i];

				temp=x[i];  x[i]=x[j];  x[j]=temp;
				i++;  j--;
			}
		}

		rho=V_i_e-j;  rho+=rho_2;  s+=s_2;
		f_lambda=s-rho*lambda-z;

		if ( fabs(f_lambda)< delta ){
			flag=1;
		}

		if (f_lambda <0){
			lambda_2=lambda; s_2=s;	rho_2=rho; f_lambda_2=f_lambda;

			V_i_e=j;  V_i=V_i_e-V_i_b+1;
		}
		else{
			lambda_1=lambda; rho_1=rho;	s_1=s; f_lambda_1=f_lambda;

			V_i_b=i; V_i=V_i_e-V_i_b+1;
		}

		if (V_i==0){
			/*printf("\n rho=%d, rho_1=%d, rho_2=%d",rho, rho_1, rho_2);

			  printf("\n V_i=%d",V_i);*/

			lambda=(s - z)/ rho;
			flag=1;
		}
		/*-------------------------------------------------------------------
		  End of initialization
		 *--------------------------------------------------------------------
		 */

	}/* end of if(!flag) */

	while (!flag){
		iter_step++;

		/* compute lambda_T  */
		lambda_T=lambda_1 + f_lambda_1 /rho_1;
		if(rho_2 !=0){
			if (lambda_2 + f_lambda_2 /rho_2 >	lambda_T)
				lambda_T=lambda_2 + f_lambda_2 /rho_2;
		}

		/* compute lambda_S */
		lambda_S=lambda_2 - f_lambda_2 *(lambda_2-lambda_1)/(f_lambda_2-f_lambda_1);

		if (fabs(lambda_T-lambda_S) <= delta){
			lambda=lambda_T; flag=1;
			break;
		}

		/* set lambda as the middle point of lambda_T and lambda_S */
		lambda=(lambda_T+lambda_S)/2;

		s_T=s_S=s=0;
		rho_T=rho_S=rho=0;
		i=V_i_b; j=V_i_e;
		while (i <= j){
			while( (i <= V_i_e) && (x[i] <= lambda) ){
				if (x[i]> lambda_T){
					s_T+=x[i]; rho_T++;
				}
				i++;
			}
			while( (j>=V_i_b) && (x[j] > lambda) ){
				if (x[j] > lambda_S){
					s_S+=x[j]; rho_S++;
				}
				else{
					s+=x[j];  rho++;
				}
				j--;
			}
			if (i<j){
				if (x[i] > lambda_S){
					s_S+=x[i]; rho_S++;
				}
				else{
					s+=x[i]; rho++;
				}

				if (x[j]> lambda_T){
					s_T+=x[j]; rho_T++;
				}

				temp=x[i]; x[i]=x[j];  x[j]=temp;
				i++; j--;
			}
		}

		s_S+=s_2; rho_S+=rho_2;
		s+=s_S; rho+=rho_S;
		s_T+=s; rho_T+=rho;
		f_lambda_S=s_S-rho_S*lambda_S-z;
		f_lambda=s-rho*lambda-z;
		f_lambda_T=s_T-rho_T*lambda_T-z;

		/*printf("\n %d & %d  & %5.6f & %5.6f & %5.6f & %5.6f & %5.6f \\\\ \n \\hline ", iter_step, V_i, lambda_1, lambda_T, lambda, lambda_S, lambda_2);*/

		if ( fabs(f_lambda)< delta ){
			/*printf("\n lambda");*/
			flag=1;
			break;
		}
		if ( fabs(f_lambda_S)< delta ){
			/* printf("\n lambda_S");*/
			lambda=lambda_S; flag=1;
			break;
		}
		if ( fabs(f_lambda_T)< delta ){
			/* printf("\n lambda_T");*/
			lambda=lambda_T; flag=1;
			break;
		}

		/*
		   printf("\n\n f_lambda_1=%5.6f, f_lambda_2=%5.6f, f_lambda=%5.6f",f_lambda_1,f_lambda_2, f_lambda);
		   printf("\n lambda_1=%5.6f, lambda_2=%5.6f, lambda=%5.6f",lambda_1, lambda_2, lambda);
		   printf("\n rho_1=%d, rho_2=%d, rho=%d ",rho_1, rho_2, rho);
		   */

		if (f_lambda <0){
			lambda_2=lambda;  s_2=s;  rho_2=rho;
			f_lambda_2=f_lambda;

			lambda_1=lambda_T; s_1=s_T; rho_1=rho_T;
			f_lambda_1=f_lambda_T;

			V_i_e=j;  i=V_i_b;
			while (i <= j){
				while( (i <= V_i_e) && (x[i] <= lambda_T) ){
					i++;
				}
				while( (j>=V_i_b) && (x[j] > lambda_T) ){
					j--;
				}
				if (i<j){
					x[j]=x[i];
					i++;   j--;
				}
			}
			V_i_b=i; V_i=V_i_e-V_i_b+1;
		}
		else{
			lambda_1=lambda;  s_1=s; rho_1=rho;
			f_lambda_1=f_lambda;

			lambda_2=lambda_S; s_2=s_S; rho_2=rho_S;
			f_lambda_2=f_lambda_S;

			V_i_b=i;  j=V_i_e;
			while (i <= j){
				while( (i <= V_i_e) && (x[i] <= lambda_S) ){
					i++;
				}
				while( (j>=V_i_b) && (x[j] > lambda_S) ){
					j--;
				}
				if (i<j){
					x[i]=x[j];
					i++;   j--;
				}
			}
			V_i_e=j; V_i=V_i_e-V_i_b+1;
		}

		if (V_i==0){
			lambda=(s - z)/ rho; flag=1;
			/*printf("\n V_i=0, lambda=%5.6f",lambda);*/
			break;
		}
	}/* end of while */


	for(i=0;i<n;i++){
		if (v[i] > lambda)
			x[i]=v[i]-lambda;
		else
			if (v[i]< -lambda)
				x[i]=v[i]+lambda;
			else
				x[i]=0;
	}
	*root=lambda;
	*steps=iter_step;
}

void  epp1(double *x, double *v, int n, double rho)
{
	int i;

	/*
	   we assume rho>=0
	   */

	for(i=0;i<n;i++){
		if (fabs(v[i])<=rho)
			x[i]=0;
		else
			if (v[i]< -rho)
				x[i]=v[i]+rho;
			else
				x[i]=v[i]-rho;
	}
}

void  epp2(double *x, double *v, int n, double rho)
{
	int i;
	double v2=0, ratio;

	/*
	   we assume rho>=0
	   */

	for(i=0; i< n; i++){
		v2+=v[i]*v[i];
	}
	v2=sqrt(v2);

	if (rho >= v2)
		for(i=0;i<n;i++)
			x[i]=0;
	else{
		ratio= (v2-rho) /v2;
		for(i=0;i<n;i++)
			x[i]=v[i]*ratio;
	}
}

void  eppInf(double *x, double * c, int * iter_step, double *v,  int n, double rho, double c0)
{
	int i, steps;

	/*
	   we assume rho>=0
	   */

	eplb(x, c, &steps, v, n, rho, c0);

	for(i=0; i< n; i++){
		x[i]=v[i]-x[i];
	}
	iter_step[0]=steps;
	iter_step[1]=0;
}

void zerofind(double *root, int * iterStep, double v, double p, double c, double x0)
{
	double x, f, fprime, p1=p-1, pp;
	int step=0;


	if (v==0){
		*root=0;	   *iterStep=0;	   return;
	}

	if (c==0){
		*root=v;	   * iterStep=0;	   return;
	}


	if ( (x0 <v) && (x0>0) )
		x=x0;
	else
		x=v;


	pp=pow(x, p1);
	f= x + c* pp -v;


	/*
	   We apply the Newton's method for solving the root
	   */
	while (1){
		step++;

		fprime=1 + c* p1 * pp / x;
		/*
		   The derivative at the current solution x
		   */

		x = x- f/fprime; /*
							The new solution is computed by the Newton method
							*/



		if (p>2){
			if (x>v){
				x=v;
			}
		}
		else{
			if ( (x<0) || (x>v)){
				x=1e-30;

				f= x+c* pow(x,p1)-v;

				if (f>0){ /*
							 If f(x) = x + c x^{p-1} - v <0 at x=1e-30,
							 this shows that the real root is between (0, 1e-30).
							 For numerical reason, we just set x=0
							 */

					*root=x;
					* iterStep=step;

					break;
				}
			}
		}
		/*
		   This makes sure that x lies in the interval [0, v]
		   */

		pp=pow(x, p1);
		f= x + c* pp -v;
		/*
		   The function value at the new solution
		   */

		if ( fabs(f) <= delta){
			*root=x;
			* iterStep=step;
			break;
		}

		if (step>=innerIter){
			printf("\n The number of steps exceed %d, in finding the root for f(x)= x + c x^{p-1} - v, 0< x< v.", innerIter);
			printf("\n If you meet with this problem, please contact Jun Liu (j.liu@asu.edu). Thanks!");
			return;
		}

	}

	/*
	   printf("\n x=%e, f=%e, step=%d\n",x, f, step);
	   */
}

double norm(double * v, double p, int n)
{
	int i;
	double t=0;


	/*
	   we assume that v[i]>=0
	   p>1
	   */

	for(i=0;i<n;i++)
		t+=pow(v[i], p);

	return( pow(t, 1/p) );
};

void eppO(double *x, double * cc, int * iter_step, double *v,  int n, double rho, double p)
{
	int i, *flag, bisStep, newtonStep=0, totoalStep=0;
	double vq=0, epsilon, vmax=0, vmin=1e10; /* we assume that the minimal value in |v| is less than 1e10*/
	double q=1/(1-1/p), c, c1, c2, root, f, xp;

	double x_diff=0; /* this value denotes the maximal difference of the x values computed from c1 and c2*/
	double temp;
	int p_n=1; /* p_n indicates the previous phi(c) is positive or negative*/

	flag=(int *)malloc(sizeof(int)*n);

	/*
	   compute vq, the q-norm of v
	   flag denotes the sign of v:
	   flag[i]=0 denotes v[i] is non-negative
	   flag[i]=1 denotes v[i] is negative
	   vmin and vmax are the maximal and minimal value of |v| (excluding 0)
	   */
	for(i=0; i< n; i++){

		x[i]=0;

		if (v[i]==0)
			flag[i]=0;
		else
		{
			if (v[i]>0)
				flag[i]=0;
			else
			{
				flag[i]=1;
				v[i]=-v[i];/*
							  we set v[i] to its absolute value
							  */
			}

			vq+=pow(v[i], q);


			if (v[i]>vmax)
				vmax=v[i];

			if (v[i]<vmin)
				vmin=v[i];
		}
	}
	vq=pow(vq, 1/q);

	/*
	   zero solution
	   */
	if (rho >= vq){
		*cc=0;
		iter_step[0]=iter_step[1]=0;


		for(i=0;i<n;i++){
			if (flag[i])
				v[i]=-v[i]; /* set the value of v[i] back*/
		}

		free(flag);
		return;
	}

	/*
	   compute epsilon
	   initialize c1 and c2, the interval where the root lies
	   */
	epsilon=(vq -rho)/ vq;
	if (p>2){

		if ( log((1-epsilon) * vmin) - (p-1) * log( epsilon* vmin ) >= 709 )
		{
			/* If this contition holds, we have c2 >= 1e308, exceeding the machine precision.

			   In this case, the reason is that p is too large
			   and meanwhile epsilon * vmin is typically small.

			   For numerical stablity, we just regard p=inf, and run eppInf
			   */


			for(i=0;i<n;i++){
				if (flag[i])
					v[i]=-v[i]; /* set the value of v[i] back*/
			}

			eppInf(x, cc, iter_step, v,  n, rho, 0);

			free(flag);
			return;
		}

		c1= (1-epsilon) * vmax / pow(epsilon* vmax, p-1);
		c2= (1-epsilon) * vmin / pow(epsilon* vmin, p-1);
	}
	else{ /*1 < p < 2*/

		c2= (1-epsilon) * vmax / pow(epsilon* vmax, p-1);
		c1= (1-epsilon) * vmin / pow(epsilon* vmin, p-1);
	}


	/*
	   printf("\n c1=%e, c2=%e", c1, c2);
	   */

	if (fabs(c1-c2) <= delta){
		c=c1;
	}
	else
		c=(c1+c2)/2;


	bisStep =0;

	while(1){
		bisStep++;

		/*compute the root corresponding to c*/
		x_diff=0;
		for(i=0;i<n;i++){
			zerofind(&root, &newtonStep, v[i], p, c, x[i]);

			temp=fabs(root-x[i]);
			if (x_diff< temp )
				x_diff=temp; /*x_diff denotes the largest gap to the previous solution*/

			x[i]=root;
			totoalStep+=newtonStep;
		}

		xp=norm(x, p, n);

		f=rho * pow(xp, 1-p) - c;

		if ( fabs(f)<=delta || fabs(c1-c2)<=delta )
			break;
		else{
			if (f>0){
				if ( (x_diff <=delta) && (p_n==0) )
					break;

				c1=c;  p_n=1;
			}
			else{

				if ( (x_diff <=delta) && (p_n==1) )
					break;

				c2=c;  p_n=0;
			}
		}
		c=(c1+c2)/2;

		if (bisStep>=outerIter){


			if ( fabs(c1-c2) <=delta * c2 )
				break;
			else{
				printf("\n The number of bisection steps exceed %d.", outerIter);
				printf("\n c1=%e, c2=%e, x_diff=%e, f=%e",c1,c2,x_diff,f);
				printf("\n If you meet with this problem, please contact Jun Liu (j.liu@asu.edu). Thanks!");

				return;
			}
		}

		/*
		   printf("\n c1=%e, c2=%e, f=%e, newtonStep=%d", c1, c2, f, newtonStep);
		   */
	}

	/*
	   printf("\n c1=%e, c2=%e, x_diff=%e, f=%e, bisStep=%d, totoalStep=%d",c1,c2, x_diff, f,bisStep,totoalStep);
	   */

	for(i=0;i<n;i++){
		if (flag[i]){
			x[i]=-x[i];
			v[i]=-v[i];
		}
	}
	free(flag);

	*cc=c;

	iter_step[0]=bisStep;
	iter_step[1]=totoalStep;
}

void epp(double *x, double * c, int * iter_step, double * v, int n, double rho, double p, double c0){
	if (rho <0){
		printf("\n rho should be non-negative!");
		exit(1);
	}

	if (p==1){
		epp1(x, v, n, rho);
		*c=0;
		iter_step[0]=iter_step[1]=0;
	}
	else
		if (p==2){
			epp2(x, v, n, rho);
			*c=0;
			iter_step[0]=iter_step[1]=0;
		}
		else
			if (p>=1e6) /* when p >=1e6, we treat it as infity*/
				eppInf(x, c, iter_step, v,  n, rho, c0);
			else
				eppO(x, c, iter_step, v,  n, rho, p);
}

