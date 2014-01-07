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

#include <lib/slep/tree/altra.h>

void altra(double *x, double *v, int n, double *ind, int nodes, double mult)
{
	int i, j;
	double lambda,twoNorm, ratio;

	/*
	 * test whether the first node is special
	 */
	if ((int) ind[0]==-1){

		/*
		 *Recheck whether ind[1] equals to zero
		 */
		if ((int) ind[1]!=-1){
			printf("\n Error! \n Check ind");
			exit(1);
		}

		lambda=mult*ind[2];

		for(j=0;j<n;j++){
			if (v[j]>lambda)
				x[j]=v[j]-lambda;
			else
				if (v[j]<-lambda)
					x[j]=v[j]+lambda;
				else
					x[j]=0;
		}

		i=1;
	}
	else{
		memcpy(x, v, sizeof(double) * n);
		i=0;
	}

	/*
	 * sequentially process each node
	 *
	 */
	for(;i < nodes; i++){
		/*
		 * compute the L2 norm of this group
		 */
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
		twoNorm=sqrt(twoNorm);

		lambda=mult*ind[3*i+2];
		if (twoNorm>lambda){
			ratio=(twoNorm-lambda)/twoNorm;

			/*
			 * shrinkage this group by ratio
			 */
			for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
				x[j]*=ratio;
		}
		else{
			/*
			 * threshold this group to zero
			 */
			for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
				x[j]=0;
		}
	}
}

void altra_mt(double *X, double *V, int n, int k, double *ind, int nodes, double mult)
{
	int i, j;

	double *x=(double *)malloc(sizeof(double)*k);
	double *v=(double *)malloc(sizeof(double)*k);

	for (i=0;i<n;i++){
		/*
		 * copy a row of V to v
		 *
		 */
		for(j=0;j<k;j++)
			v[j]=V[j*n + i];

		altra(x, v, k, ind, nodes, mult);

		/*
		 * copy the solution to X
		 */
		for(j=0;j<k;j++)
			X[j*n+i]=x[j];
	}

	free(x);
	free(v);
}

void computeLambda2Max(double *lambda2_max, double *x, int n, double *ind, int nodes)
{
	int i, j;
	double twoNorm;

	*lambda2_max=0;

	for(i=0;i < nodes; i++){
		/*
		 * compute the L2 norm of this group
		 */
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
		twoNorm=sqrt(twoNorm);

		twoNorm=twoNorm/ind[3*i+2];

		if (twoNorm >*lambda2_max )
			*lambda2_max=twoNorm;
	}
}

double treeNorm(double *x, int ldx, int n, double *ind, int nodes){

	int i, j;
	double twoNorm, lambda;

	double tree_norm = 0;

	/*
	 * test whether the first node is special
	 */
	if ((int) ind[0]==-1){

		/*
		 *Recheck whether ind[1] equals to zero
		 */
		if ((int) ind[1]!=-1){
			printf("\n Error! \n Check ind");
			exit(1);
		}

		lambda=ind[2];

		for(j=0;j<n*ldx;j+=ldx){
			tree_norm+=fabs(x[j]);
		}

		tree_norm = tree_norm * lambda;

		i=1;
	}
	else{
		i=0;
	}

	/*
	 * sequentially process each node
	 *
	 */
	for(;i < nodes; i++){
		/*
		 * compute the L2 norm of this group
		 */
		twoNorm=0;

		int n_in_node = (int) ind[3*i+1] - (int) ind[3*i]-1;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i]-1 + n_in_node*ldx;j+=ldx)
			twoNorm += x[j] * x[j];
		twoNorm=sqrt(twoNorm);

		lambda=ind[3*i+2];

		tree_norm = tree_norm + lambda*twoNorm;
	}

	return tree_norm;
}

double findLambdaMax(double *v, int n, double *ind, int nodes){

	int i;
	double lambda=0,squaredWeight=0, lambda1,lambda2;
	double *x=(double *)malloc(sizeof(double)*n);
	double *ind2=(double *)malloc(sizeof(double)*nodes*3);
	int num=0;

	for(i=0;i<n;i++){
		lambda+=v[i]*v[i];
	}

	if ( (int)ind[0]==-1 )
		squaredWeight=n*ind[2]*ind[2];
	else
		squaredWeight=ind[2]*ind[2];

	for (i=1;i<nodes;i++){
		squaredWeight+=ind[3*i+2]*ind[3*i+2];
	}

	/* set lambda to an initial guess
	*/
	lambda=sqrt(lambda/squaredWeight);

	/*
	   printf("\n\n   lambda=%2.5f",lambda);
	   */

	/*
	 *copy ind to ind2,
	 *and scale the weight 3*i+2
	 */
	for(i=0;i<nodes;i++){
		ind2[3*i]=ind[3*i];
		ind2[3*i+1]=ind[3*i+1];
		ind2[3*i+2]=ind[3*i+2]*lambda;
	}

	/* test whether the solution is zero or not
	*/
	altra(x, v, n, ind2, nodes);
	for(i=0;i<n;i++){
		if (x[i]!=0)
			break;
	}

	if (i>=n) {
		/*x is a zero vector*/
		lambda2=lambda;
		lambda1=lambda;

		num=0;

		while(1){
			num++;

			lambda2=lambda;
			lambda1=lambda1/2;
			/* update ind2
			*/
			for(i=0;i<nodes;i++){
				ind2[3*i+2]=ind[3*i+2]*lambda1;
			}

			/* compute and test whether x is zero
			*/
			altra(x, v, n, ind2, nodes);
			for(i=0;i<n;i++){
				if (x[i]!=0)
					break;
			}

			if (i<n){
				break;
				/*x is not zero
				 *we have found lambda1
				 */
			}
		}

	}
	else{
		/*x is a non-zero vector*/
		lambda2=lambda;
		lambda1=lambda;

		num=0;
		while(1){
			num++;

			lambda1=lambda2;
			lambda2=lambda2*2;
			/* update ind2
			*/
			for(i=0;i<nodes;i++){
				ind2[3*i+2]=ind[3*i+2]*lambda2;
			}

			/* compute and test whether x is zero
			*/
			altra(x, v, n, ind2, nodes);
			for(i=0;i<n;i++){
				if (x[i]!=0)
					break;
			}

			if (i>=n){
				break;
				/*x is a zero vector
				 *we have found lambda2
				 */
			}
		}
	}

	/*
	   printf("\n num=%d, lambda1=%2.5f, lambda2=%2.5f",num, lambda1,lambda2);
	   */

	while ( fabs(lambda2-lambda1) > lambda2 * 1e-10 ){

		num++;

		lambda=(lambda1+lambda2)/2;

		/* update ind2
		*/
		for(i=0;i<nodes;i++){
			ind2[3*i+2]=ind[3*i+2]*lambda;
		}

		/* compute and test whether x is zero
		*/
		altra(x, v, n, ind2, nodes);
		for(i=0;i<n;i++){
			if (x[i]!=0)
				break;
		}

		if (i>=n){
			lambda2=lambda;
		}
		else{
			lambda1=lambda;
		}

		/*
		   printf("\n lambda1=%2.5f, lambda2=%2.5f",lambda1,lambda2);
		   */
	}

	/*
	   printf("\n num=%d",num);

	   printf("   lambda1=%2.5f, lambda2=%2.5f",lambda1,lambda2);

*/

	free(x);
	free(ind2);

	return lambda2;
}

double findLambdaMax_mt(double *V, int n, int k, double *ind, int nodes)
{
	int i, j;

	double *v=(double *)malloc(sizeof(double)*k);
	double lambda;

	double lambdaMax=0;

	for (i=0;i<n;i++){
		/*
		 * copy a row of V to v
		 *
		 */
		for(j=0;j<k;j++)
			v[j]=V[j*n + i];

		lambda = findLambdaMax(v, k, ind, nodes);

		/*
		   printf("\n   lambda=%5.2f",lambda);
		   */

		if (lambda>lambdaMax)
			lambdaMax=lambda;
	}

	/*
	   printf("\n *lambdaMax=%5.2f",*lambdaMax);
	   */

	free(v);
	return lambdaMax;
}
