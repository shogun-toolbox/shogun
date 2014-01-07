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

#include <lib/slep/flsa/flsa.h>
#include <lib/slep/flsa/sfa.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void flsa(double *x, double *z, double *infor,
		double * v, double *z0,
		double lambda1, double lambda2, int n,
		int maxStep, double tol, int tau, int flag)
{

	int i, nn=n-1, m;
	double zMax, temp;
	double *Av, *g, *s;
	int iterStep, numS;
	double gap;
	double *zz = NULL; /*to replace z0, so that z0 shall not revised after */


	Av=(double *) malloc(sizeof(double)*nn);

	/*
	   Compute Av= A*v                  (n=4, nn=3)
	   A= [ -1  1  0  0;
	   0  -1 1  0;
	   0  0  -1 1]
	   */

	for (i=0;i<nn; i++)
		Av[i]=v[i+1]-v[i];

	/*
	   Sovlve the linear system via Thomas's algorithm or Rose's algorithm
	   B * z0 = Av
	   */

	Thomas(&zMax, z, Av, nn);

	/*
	   We consider two cases:
	   1) lambda2 >= zMax, which leads to a solution with same entry values
	   2) lambda2 < zMax, which needs to first run sfa, and then perform soft thresholding
	   */

	/*
	   First case: lambda2 >= zMax
	   */
	if (lambda2 >= zMax){

		temp=0;
		m=n%5;
		if (m!=0){
			for (i=0;i<m;i++)
				temp+=v[i];
		}
		for (i=m;i<n;i+=5){
			temp += v[i] + v[i+1] + v[i+2] + v[i+3] + v[i+4];
		}
		temp/=n;
		/* temp is the mean value of v*/


		/*
		   soft thresholding by lambda1
		   */
		if (temp> lambda1)
			temp= temp-lambda1;
		else
			if (temp < -lambda1)
				temp= temp+lambda1;
			else
				temp=0;

		m=n%7;
		if (m!=0){
			for (i=0;i<m;i++)
				x[i]=temp;
		}
		for (i=m;i<n;i+=7){
			x[i]   =temp;
			x[i+1] =temp;
			x[i+2] =temp;
			x[i+3] =temp;
			x[i+4] =temp;
			x[i+5] =temp;
			x[i+6] =temp;
		}

		gap=0;

		free(Av);

		if (infor)
		{
			infor[0]= gap;
			infor[1]= 0;
			infor[2]=zMax;
			infor[3]=0;
		}

		return;
	}


	/*
	   Second case: lambda2 < zMax

	   We need to call sfa for computing x, and then do soft thresholding

	   Before calling sfa, we need to allocate memory for g and s,
	   and initialize z and z0.
	   */


	/*
	   Allocate memory for g and s
	   */

	g    =(double *) malloc(sizeof(double)*nn),
		 s    =(double *) malloc(sizeof(double)*nn);



	m=flag /10;
	/*

	   If m=0, then this shows that, z0 is a "good" starting point. (m=1-6)

	   Otherwise (m=11-16), we shall set z as either the solution to the linear system.
	   or the zero point

*/
	if (m==0){
		for (i=0;i<nn;i++){
			if (z0[i] > lambda2)
				z[i]=lambda2;
			else
				if (z0[i]<-lambda2)
					z[i]=-lambda2;
				else
					z[i]=z0[i];
		}
	}
	else{
		if (lambda2 >= 0.5 * zMax){
			for (i=0;i<nn;i++){
				if (z[i] > lambda2)
					z[i]=lambda2;
				else
					if (z[i]<-lambda2)
						z[i]=-lambda2;
			}
		}
		else{
			for (i=0;i<nn;i++)
				z[i]=0;

		}
	}

	flag=flag %10;  /*
					   flag is now in [1:6]

					   for sfa, i.e., flag in [1:4], we need initialize z0 with zero
					   */

	if (flag>=1 && flag<=4){
		zz    =(double *) malloc(sizeof(double)*nn);

		for (i=0;i<nn;i++)
			zz[i]=0;
	}

	/*
	   call sfa, sfa_one, or sfa_special to compute z, for finding the subgradient
	   and x
	   */

	if (flag==6)
		iterStep=sfa_one(x, &gap, &numS,
				z,  v,   Av,
				lambda2, nn,  maxStep,
				s, g,
				tol, tau);
	else
		if (flag==5)
			iterStep=sfa_special(x, &gap, &numS,
					z,  v,   Av,
					lambda2, nn,  maxStep,
					s, g,
					tol, tau);
		else{
			iterStep=sfa(x, &gap, &numS,
					z, zz,   v,  Av,
					lambda2, nn, maxStep,
					s,  g,
					tol,tau, flag);

			free (zz);
			/*free the variable zz*/
		}


	/*
	   soft thresholding by lambda1
	   */

	for(i=0;i<n;i++)
		if (x[i] > lambda1)
			x[i]-=lambda1;
		else
			if (x[i]<-lambda1)
				x[i]+=lambda1;
			else
				x[i]=0;


	free(Av);
	free(g);
	free(s);

	if (infor)
	{
		infor[0]=gap;
		infor[1]=iterStep;
		infor[2]=zMax;
		infor[3]=numS;
	}
}

