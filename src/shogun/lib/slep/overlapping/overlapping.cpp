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

#include <lib/slep/overlapping/overlapping.h>

void identifySomeZeroEntries(double * u, int * zeroGroupFlag, int *entrySignFlag,
		int *pp, int *gg,
		double *v, double lambda1, double lambda2,
		int p, int g, double * w, double *G){

	int i, j, newZeroNum, iterStep=0;
	double twoNorm, temp;

	/*
	 * process the L1 norm
	 *
	 * generate the u>=0, and assign values to entrySignFlag
	 *
	 */
	for(i=0;i<p;i++){
		if (v[i]> lambda1){
			u[i]=v[i]-lambda1;

			entrySignFlag[i]=1;
		}
		else{
			if (v[i] < -lambda1){
				u[i]= -v[i] -lambda1;

				entrySignFlag[i]=-1;
			}
			else{
				u[i]=0;

				entrySignFlag[i]=0;
			}
		}
	}

	/*
	 * Applying Algorithm 1 for identifying some sparse groups
	 *
	 */

	/* zeroGroupFlag denotes whether the corresponding group is zero */
	for(i=0;i<g;i++)
		zeroGroupFlag[i]=1;

	while(1){

		iterStep++;

		if (iterStep>g+1){

			printf("\n Identify Zero Group: iterStep= %d. The code might have a bug! Check it!", iterStep);
			return;
		}

		/*record the number of newly detected sparse groups*/
		newZeroNum=0;

		for (i=0;i<g;i++){

			if(zeroGroupFlag[i]){

				/*compute the two norm of the */

				twoNorm=0;
				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
					temp=u[ (int) G[j]];
					twoNorm+=temp*temp;
				}
				twoNorm=sqrt(twoNorm);

				/*
				   printf("\n twoNorm=%2.5f, %2.5f",twoNorm,lambda2 * w[3*i+2]);
				   */

				/*
				 * Test whether this group should be sparse
				 */
				if (twoNorm<= lambda2 * w[3*i+2] ){
					zeroGroupFlag[i]=0;

					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
						u[ (int) G[j]]=0;

					newZeroNum++;

					/*
					   printf("\n zero group=%d", i);
					   */
				}
			} /*end of if(!zeroGroupFlag[i]) */

		} /*end of for*/

		if (newZeroNum==0)
			break;
	}

	*pp=0;
	/* zeroGroupFlag denotes whether the corresponding entry is zero */
	for(i=0;i<p;i++){
		if (u[i]==0){
			entrySignFlag[i]=0;
			*pp=*pp+1;
		}
	}

	*gg=0;
	for(i=0;i<g;i++){
		if (zeroGroupFlag[i]==0)
			*gg=*gg+1;
	}
}

void xFromY(double *x, double *y,
		double *u, double *Y,
		int p, int g, int *zeroGroupFlag,
		double *G, double *w){

	int i,j;


	for(i=0;i<p;i++)
		x[i]=u[i];

	for(i=0;i<g;i++){
		if(zeroGroupFlag[i]){ /*this group is non-zero*/

			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
				x[ (int) G[j] ] -= Y[j];
			}
		}
	}/*end of for(i=0;i<g;i++) */

	for(i=0;i<p;i++){
		if (x[i]>=0){
			y[i]=0;
		}
		else{
			y[i]=x[i];
			x[i]=0;
		}
	}
}

void YFromx(double *Y,
		double *xnew, double *Ynew,
		double lambda2, int g, int *zeroGroupFlag,
		double *G, double *w){

	int i, j;
	double twoNorm, temp;

	for(i=0;i<g;i++){
		if(zeroGroupFlag[i]){ /*this group is non-zero*/

			twoNorm=0;
			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
				temp=xnew[ (int) G[j] ];

				Y[j]=temp;

				twoNorm+=temp*temp;
			}
			twoNorm=sqrt(twoNorm); /* two norm for x_{G_i}*/

			if (twoNorm > 0 ){ /*if x_{G_i} is non-zero*/
				temp=lambda2 * w[3*i+2] / twoNorm;

				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
					Y[j] *= temp;
			}
			else  /*if x_{G_j} =0, we let Y^i=Ynew^i*/
			{
				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
					Y[j]=Ynew[j];
			}
		}
	}/*end of for(i=0;i<g;i++) */
}

void dualityGap(double *gap, double *penalty2,
		double *x, double *Y, int g, int *zeroGroupFlag,
		double *G, double *w, double lambda2){

	int i,j;
	double temp, twoNorm, innerProduct;

	*gap=0; *penalty2=0;

	for(i=0;i<g;i++){
		if(zeroGroupFlag[i]){ /*this group is non-zero*/

			twoNorm=0;innerProduct=0;

			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
				temp=x[ (int) G[j] ];

				twoNorm+=temp*temp;

				innerProduct+=temp * Y[j];
			}

			twoNorm=sqrt(twoNorm)* w[3*i +2];

			*penalty2+=twoNorm;

			twoNorm=lambda2 * twoNorm;
			if (twoNorm > innerProduct)
				*gap+=twoNorm-innerProduct;
		}
	}/*end of for(i=0;i<g;i++) */
}

void overlapping_gd(double *x, double *gap, double *penalty2,
		double *v, int p, int g, double lambda1, double lambda2,
		double *w, double *G, double *Y, int maxIter, int flag, double tol){

	int YSize=(int) w[3*(g-1) +1]+1;
	double *u=(double *)malloc(sizeof(double)*p);
	double *y=(double *)malloc(sizeof(double)*p);

	double *xnew=(double *)malloc(sizeof(double)*p);
	double *Ynew=(double *)malloc(sizeof(double)* YSize );

	int *zeroGroupFlag=(int *)malloc(sizeof(int)*g);
	int *entrySignFlag=(int *)malloc(sizeof(int)*p);
	int pp, gg;
	int i, j, iterStep;
	double twoNorm,temp, L=1, leftValue, rightValue, gapR, penalty2R;
	int nextRestartStep=0;

	/*
	 * call the function to identify some zero entries
	 *
	 * entrySignFlag[i]=0 denotes that the corresponding entry is definitely zero
	 *
	 * zeroGroupFlag[i]=0 denotes that the corresponding group is definitely zero
	 *
	 */

	identifySomeZeroEntries(u, zeroGroupFlag, entrySignFlag,
			&pp, &gg,
			v, lambda1, lambda2,
			p, g, w, G);

	penalty2[1]=pp;
	penalty2[2]=gg;
	/*store pp and gg to penalty2[1] and penalty2[2]*/


	/*
	 *-------------------
	 *  Process Y
	 *-------------------
	 * We make sure that Y is feasible
	 *    and if x_i=0, then set Y_{ij}=0
	 */
	for(i=0;i<g;i++){

		if(zeroGroupFlag[i]){ /*this group is non-zero*/

			/*compute the two norm of the group*/
			twoNorm=0;

			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){

				if (! u[ (int) G[j] ] )
					Y[j]=0;

				twoNorm+=Y[j]*Y[j];
			}
			twoNorm=sqrt(twoNorm);

			if (twoNorm > lambda2 * w[3*i+2] ){
				temp=lambda2 * w[3*i+2] / twoNorm;

				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
					Y[j]*=temp;
			}
		}
		else{ /*this group is zero*/
			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
				Y[j]=0;
		}
	}

	/*
	 * set Ynew to zero
	 *
	 * in the following processing, we only operator Y and Ynew in the
	 * possibly non-zero groups by "if(zeroGroupFlag[i])"
	 *
	 */
	for(i=0;i<YSize;i++)
		Ynew[i]=0;

	/*
	 * ------------------------------------
	 * Gradient Descent begins here
	 * ------------------------------------
	 */

	/*
	 * compute x=max(u-Y * e, 0);
	 *
	 */
	xFromY(x, y, u, Y, p, g, zeroGroupFlag, G, w);


	/*the main loop */

	for(iterStep=0;iterStep<maxIter;iterStep++){


		/*
		 * the gradient at Y is
		 *
		 *   omega'(Y)=-x e^T
		 *
		 *  where  x=max(u-Y * e, 0);
		 *
		 */


		/*
		 * line search to find Ynew with appropriate L
		 */

		while (1){
			/*
			 * compute
			 * Ynew = proj ( Y + x e^T / L )
			 */
			for(i=0;i<g;i++){
				if(zeroGroupFlag[i]){ /*this group is non-zero*/

					twoNorm=0;
					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
						Ynew[j]= Y[j] + x[ (int) G[j] ] / L;

						twoNorm+=Ynew[j]*Ynew[j];
					}
					twoNorm=sqrt(twoNorm);

					if (twoNorm > lambda2 * w[3*i+2] ){
						temp=lambda2 * w[3*i+2] / twoNorm;

						for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
							Ynew[j]*=temp;
					}
				}
			}/*end of for(i=0;i<g;i++) */

			/*
			 * compute xnew=max(u-Ynew * e, 0);
			 *
			 *void xFromY(double *x, double *y,
			 *            double *u, double *Y,
			 *            int p, int g, int *zeroGroupFlag,
			 *            double *G, double *w)
			 */
			xFromY(xnew, y, u, Ynew, p, g, zeroGroupFlag, G, w);

			/* test whether L is appropriate*/
			leftValue=0;
			for(i=0;i<p;i++){
				if (entrySignFlag[i]){
					temp=xnew[i]-x[i];

					leftValue+= temp * ( 0.5 * temp + y[i]);
				}
			}

			rightValue=0;
			for(i=0;i<g;i++){
				if(zeroGroupFlag[i]){ /*this group is non-zero*/

					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
						temp=Ynew[j]-Y[j];

						rightValue+=temp * temp;
					}
				}
			}/*end of for(i=0;i<g;i++) */
			rightValue=rightValue/2;

			if ( leftValue <= L * rightValue){

				temp= L * rightValue / leftValue;

				if (temp >5)
					L=L*0.8;

				break;
			}
			else{
				temp=leftValue / rightValue;

				if (L*2 <= temp)
					L=temp;
				else
					L=2*L;


				if ( L / g - 2* g ){

					if (rightValue < 1e-16){
						break;
					}
					else{

						printf("\n GD: leftValue=%e, rightValue=%e, ratio=%e", leftValue, rightValue, temp);

						printf("\n L=%e > 2 * %d * %d. There might be a bug here. Otherwise, it is due to numerical issue.", L, g, g);

						break;
					}
				}
			}
		}

		/* compute the duality gap at (xnew, Ynew)
		 *
		 * void dualityGap(double *gap, double *penalty2,
		 *               double *x, double *Y, int g, int *zeroGroupFlag,
		 *               double *G, double *w, double lambda2)
		 *
		 */
		dualityGap(gap, penalty2, xnew, Ynew, g, zeroGroupFlag, G, w, lambda2);

		/*
		 * flag =1 means restart
		 *
		 * flag =0 means with restart
		 *
		 * nextRestartStep denotes the next "step number" for
		 *            initializing the restart process.
		 *
		 * This is based on the fact that, the result is only beneficial when
		 *    xnew is good. In other words,
		 *             if xnew is not good, then the
		 *                restart might not be helpful.
		 */

		if ( (flag==0) || (flag==1 && iterStep < nextRestartStep )){

			/* copy Ynew to Y, and xnew to x */
			memcpy(x, xnew, sizeof(double) * p);
			memcpy(Y, Ynew, sizeof(double) * YSize);

			/*
			   printf("\n iterStep=%d, L=%2.5f, gap=%e", iterStep, L, *gap);
			   */

		}
		else{
			/*
			 * flag=1
			 *
			 * We allow the restart of the program.
			 *
			 * Here, Y is constructed as a subgradient of xnew, based on the
			 *   assumption that Y might be a better choice than Ynew, provided
			 *   that xnew is good enough.
			 *
			 */

			/*
			 * compute the restarting point Y with xnew and Ynew
			 *
			 *void YFromx(double *Y,
			 *            double *xnew, double *Ynew,
			 *            double lambda2, int g, int *zeroGroupFlag,
			 *            double *G, double *w)
			 */
			YFromx(Y, xnew, Ynew, lambda2, g, zeroGroupFlag, G, w);

			/*compute the solution with the starting point Y
			 *
			 *void xFromY(double *x, double *y,
			 *            double *u, double *Y,
			 *            int p, int g, int *zeroGroupFlag,
			 *            double *G, double *w)
			 *
			 */
			xFromY(x, y, u, Y, p, g, zeroGroupFlag, G, w);

			/*compute the duality at (x, Y)
			 *
			 * void dualityGap(double *gap, double *penalty2,
			 *               double *x, double *Y, int g, int *zeroGroupFlag,
			 *               double *G, double *w, double lambda2)
			 *
			 */
			dualityGap(&gapR, &penalty2R, x, Y, g, zeroGroupFlag, G, w, lambda2);

			if (*gap< gapR){
				/*(xnew, Ynew) is better in terms of duality gap*/
				/* copy Ynew to Y, and xnew to x */
				memcpy(x, xnew, sizeof(double) * p);
				memcpy(Y, Ynew, sizeof(double) * YSize);

				/*In this case, we do not apply restart, as (x,Y) is not better
				 *
				 * We postpone the "restart" by giving a
				 *           "nextRestartStep"
				 */

				/*
				 * we test *gap here, in case *gap=0
				 */
				if (*gap <=tol)
					break;
				else{
					nextRestartStep=iterStep+ (int) sqrt(gapR / *gap);
				}
			}
			else{ /*we use (x, Y), as it is better in terms of duality gap*/
				*gap=gapR;
				*penalty2=penalty2R;
			}

			/*
			   printf("\n iterStep=%d, L=%2.5f, gap=%e, gapR=%e", iterStep, L, *gap, gapR);
			   */

		}

		/*
		 * if the duality gap is within pre-specified parameter tol
		 *
		 * we terminate the algorithm
		 */
		if (*gap <=tol)
			break;
	}

	penalty2[3]=iterStep;

	penalty2[4]=0;
	for(i=0;i<g;i++){
		if (zeroGroupFlag[i]==0)
			penalty2[4]=penalty2[4]+1;
		else{
			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
				if (x[ (int) G[j] ] !=0)
					break;
			}

			if (j>(int) w[3*i +1])
				penalty2[4]=penalty2[4]+1;
		}
	}

	/*
	 * assign sign to the solution x
	 */
	for(i=0;i<p;i++){
		if (entrySignFlag[i]==-1){
			x[i]=-x[i];
		}
	}

	free (u);
	free (y);
	free (xnew);
	free (Ynew);
	free (zeroGroupFlag);
	free (entrySignFlag);
}

void gradientDescentStep(double *xnew, double *Ynew,
		double *LL, double *u, double *y, int *entrySignFlag, double lambda2,
		double *x, double *Y, int p, int g, int * zeroGroupFlag,
		double *G, double *w){

	double twoNorm, temp, L=*LL, leftValue, rightValue;
	int i,j;



	while (1){

		/*
		 * compute
		 * Ynew = proj ( Y + x e^T / L )
		 */
		for(i=0;i<g;i++){
			if(zeroGroupFlag[i]){ /*this group is non-zero*/

				twoNorm=0;
				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
					Ynew[j]= Y[j] + x[ (int) G[j] ] / L;

					twoNorm+=Ynew[j]*Ynew[j];
				}
				twoNorm=sqrt(twoNorm);

				if (twoNorm > lambda2 * w[3*i+2] ){
					temp=lambda2 * w[3*i+2] / twoNorm;

					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
						Ynew[j]*=temp;
				}
			}
		}/*end of for(i=0;i<g;i++) */

		/*
		 * compute xnew=max(u-Ynew * e, 0);
		 *
		 *void xFromY(double *x, double *y,
		 *            double *u, double *Y,
		 *            int p, int g, int *zeroGroupFlag,
		 *            double *G, double *w)
		 */
		xFromY(xnew, y, u, Ynew, p, g, zeroGroupFlag, G, w);

		/* test whether L is appropriate*/
		leftValue=0;
		for(i=0;i<p;i++){
			if (entrySignFlag[i]){
				temp=xnew[i]-x[i];

				leftValue+= temp * ( 0.5 * temp + y[i]);
			}
		}

		rightValue=0;
		for(i=0;i<g;i++){
			if(zeroGroupFlag[i]){ /*this group is non-zero*/

				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
					temp=Ynew[j]-Y[j];

					rightValue+=temp * temp;
				}
			}
		}/*end of for(i=0;i<g;i++) */
		rightValue=rightValue/2;

		/*
		   printf("\n leftValue =%e, rightValue=%e, L=%e", leftValue, rightValue, L);
		   */

		if ( leftValue <= L * rightValue){

			temp= L * rightValue / leftValue;

			if (temp >5)
				L=L*0.8;

			break;
		}
		else{
			temp=leftValue / rightValue;

			if (L*2 <= temp)
				L=temp;
			else
				L=2*L;

			if ( L / g - 2* g >0 ){

				if (rightValue < 1e-16){
					break;
				}
				else{

					printf("\n One Gradient Step: leftValue=%e, rightValue=%e, ratio=%e", leftValue, rightValue, temp);

					printf("\n L=%e > 2 * %d * %d. There might be a bug here. Otherwise, it is due to numerical issue.", L, g, g);

					break;
				}
			}
		}
	}

	*LL=L;
}

void overlapping_agd(double *x, double *gap, double *penalty2,
		double *v, int p, int g, double lambda1, double lambda2,
		double *w, double *G, double *Y, int maxIter, int flag, double tol){

	int YSize=(int) w[3*(g-1) +1]+1;
	double *u=(double *)malloc(sizeof(double)*p);
	double *y=(double *)malloc(sizeof(double)*p);

	double *xnew=(double *)malloc(sizeof(double)*p);
	double *Ynew=(double *)malloc(sizeof(double)* YSize );

	double *xS=(double *)malloc(sizeof(double)*p);
	double *YS=(double *)malloc(sizeof(double)* YSize );

	/*double *xp=(double *)malloc(sizeof(double)*p);*/
	double *Yp=(double *)malloc(sizeof(double)* YSize );

	int *zeroGroupFlag=(int *)malloc(sizeof(int)*g);
	int *entrySignFlag=(int *)malloc(sizeof(int)*p);

	int pp, gg;
	int i, j, iterStep;
	double twoNorm,temp, L=1, leftValue, rightValue, gapR, penalty2R;
	int nextRestartStep=0;

	double alpha, alphap=0.5, beta, gamma;

	/*
	 * call the function to identify some zero entries
	 *
	 * entrySignFlag[i]=0 denotes that the corresponding entry is definitely zero
	 *
	 * zeroGroupFlag[i]=0 denotes that the corresponding group is definitely zero
	 *
	 */

	identifySomeZeroEntries(u, zeroGroupFlag, entrySignFlag,
			&pp, &gg,
			v, lambda1, lambda2,
			p, g, w, G);

	penalty2[1]=pp;
	penalty2[2]=gg;
	/*store pp and gg to penalty2[1] and penalty2[2]*/

	/*
	 *-------------------
	 *  Process Y
	 *-------------------
	 * We make sure that Y is feasible
	 *    and if x_i=0, then set Y_{ij}=0
	 */
	for(i=0;i<g;i++){

		if(zeroGroupFlag[i]){ /*this group is non-zero*/

			/*compute the two norm of the group*/
			twoNorm=0;

			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){

				if (! u[ (int) G[j] ] )
					Y[j]=0;

				twoNorm+=Y[j]*Y[j];
			}
			twoNorm=sqrt(twoNorm);

			if (twoNorm > lambda2 * w[3*i+2] ){
				temp=lambda2 * w[3*i+2] / twoNorm;

				for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
					Y[j]*=temp;
			}
		}
		else{ /*this group is zero*/
			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
				Y[j]=0;
		}
	}

	/*
	 * set Ynew and Yp to zero
	 *
	 * in the following processing, we only operate, Yp, Y and Ynew in the
	 * possibly non-zero groups by "if(zeroGroupFlag[i])"
	 *
	 */
	for(i=0;i<YSize;i++)
		YS[i]=Yp[i]=Ynew[i]=0;


	/*
	 * ---------------
	 *
	 * we first do a gradient descent step for determing the value of an approporate L
	 *
	 * Also, we initialize gamma
	 *
	 * with Y, we compute a new Ynew
	 *
	 */


	/*
	 * compute x=max(u-Y * e, 0);
	 */
	xFromY(x, y, u, Y, p, g, zeroGroupFlag, G, w);

	/*
	 * compute (xnew, Ynew) from (x, Y)
	 *
	 *
	 * gradientDescentStep(double *xnew, double *Ynew,
	 double *LL, double *u, double *y, int *entrySignFlag, double lambda2,
	 double *x, double *Y, int p, int g, int * zeroGroupFlag,
	 double *G, double *w)
	 */

	gradientDescentStep(xnew, Ynew,
			&L, u, y,entrySignFlag,lambda2,
			x, Y, p, g, zeroGroupFlag,
			G, w);

	/*
	 * we have finished one gradient descent to get
	 *
	 * (x, Y) and (xnew, Ynew), where (xnew, Ynew) is
	 *
	 *    a gradient descent step based on (x, Y)
	 *
	 * we set (xp, Yp)=(x, Y)
	 *
	 *        (x, Y)= (xnew, Ynew)
	 */

	/*memcpy(xp, x, sizeof(double) * p);*/
	memcpy(Yp, Y, sizeof(double) * YSize);

	/*memcpy(x, xnew, sizeof(double) * p);*/
	memcpy(Y, Ynew, sizeof(double) * YSize);

	gamma=L;

	/*
	 * ------------------------------------
	 * Accelerated Gradient Descent begins here
	 * ------------------------------------
	 */


	for(iterStep=0;iterStep<maxIter;iterStep++){


		while (1){


			/*
			 * compute alpha as the positive root of
			 *
			 *     L * alpha^2 = (1-alpha) * gamma
			 *
			 */

			alpha= ( - gamma + sqrt( gamma * gamma + 4 * L * gamma ) ) / 2 / L;

			beta= gamma * (1-alphap)/ alphap / (gamma + L * alpha);

			/*
			 * compute YS= Y + beta * (Y - Yp)
			 *
			 */
			for(i=0;i<g;i++){
				if(zeroGroupFlag[i]){ /*this group is non-zero*/

					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){

						YS[j]=Y[j] + beta * (Y[j]-Yp[j]);

					}
				}
			}/*end of for(i=0;i<g;i++) */


			/*
			 * compute xS
			 */
			xFromY(xS, y, u, YS, p, g, zeroGroupFlag, G, w);


			/*
			 *
			 * Ynew = proj ( YS + xS e^T / L )
			 *
			 */
			for(i=0;i<g;i++){
				if(zeroGroupFlag[i]){ /*this group is non-zero*/

					twoNorm=0;
					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){

						Ynew[j]= YS[j] + xS[ (int) G[j] ] / L;

						twoNorm+=Ynew[j]*Ynew[j];
					}
					twoNorm=sqrt(twoNorm);

					if (twoNorm > lambda2 * w[3*i+2] ){
						temp=lambda2 * w[3*i+2] / twoNorm;

						for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++)
							Ynew[j]*=temp;
					}
				}
			}/*end of for(i=0;i<g;i++) */

			/*
			 * compute xnew=max(u-Ynew * e, 0);
			 *
			 *void xFromY(double *x, double *y,
			 *            double *u, double *Y,
			 *            int p, int g, int *zeroGroupFlag,
			 *            double *G, double *w)
			 */

			xFromY(xnew, y, u, Ynew, p, g, zeroGroupFlag, G, w);

			/* test whether L is appropriate*/
			leftValue=0;
			for(i=0;i<p;i++){
				if (entrySignFlag[i]){
					temp=xnew[i]-xS[i];

					leftValue+= temp * ( 0.5 * temp + y[i]);
				}
			}

			rightValue=0;
			for(i=0;i<g;i++){
				if(zeroGroupFlag[i]){ /*this group is non-zero*/

					for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
						temp=Ynew[j]-YS[j];

						rightValue+=temp * temp;
					}
				}
			}/*end of for(i=0;i<g;i++) */
			rightValue=rightValue/2;

			if ( leftValue <= L * rightValue){

				temp= L * rightValue / leftValue;

				if (temp >5)
					L=L*0.8;

				break;
			}
			else{
				temp=leftValue / rightValue;

				if (L*2 <= temp)
					L=temp;
				else
					L=2*L;



				if ( L / g - 2* g  >0 ){

					if (rightValue < 1e-16){
						break;
					}
					else{

						printf("\n AGD: leftValue=%e, rightValue=%e, ratio=%e", leftValue, rightValue, temp);

						printf("\n L=%e > 2 * %d * %d. There might be a bug here. Otherwise, it is due to numerical issue.", L, g, g);

						break;
					}
				}
			}
		}

		/* compute the duality gap at (xnew, Ynew)
		 *
		 * void dualityGap(double *gap, double *penalty2,
		 *               double *x, double *Y, int g, int *zeroGroupFlag,
		 *               double *G, double *w, double lambda2)
		 *
		 */
		dualityGap(gap, penalty2,
				xnew, Ynew, g, zeroGroupFlag,
				G, w, lambda2);


		/*
		 * if the duality gap is within pre-specified parameter tol
		 *
		 * we terminate the algorithm
		 */
		if (*gap <=tol){

			memcpy(x, xnew, sizeof(double) * p);
			memcpy(Y, Ynew, sizeof(double) * YSize);

			break;
		}



		/*
		 * flag =1 means restart
		 *
		 * flag =0 means with restart
		 *
		 * nextRestartStep denotes the next "step number" for
		 *            initializing the restart process.
		 *
		 * This is based on the fact that, the result is only beneficial when
		 *    xnew is good. In other words,
		 *             if xnew is not good, then the
		 *                restart might not be helpful.
		 */

		if ( (flag==0) || (flag==1 && iterStep < nextRestartStep )){


			/*memcpy(xp, x, sizeof(double) * p);*/
			memcpy(Yp, Y, sizeof(double) * YSize);

			/*memcpy(x, xnew, sizeof(double) * p);*/
			memcpy(Y, Ynew, sizeof(double) * YSize);

			gamma=gamma * (1-alpha);

			alphap=alpha;

			/*
			   printf("\n iterStep=%d, L=%2.5f, gap=%e", iterStep, L, *gap);
			   */

		}
		else{
			/*
			 * flag=1
			 *
			 * We allow the restart of the program.
			 *
			 * Here, Y is constructed as a subgradient of xnew, based on the
			 *   assumption that Y might be a better choice than Ynew, provided
			 *   that xnew is good enough.
			 *
			 */

			/*
			 * compute the restarting point YS with xnew and Ynew
			 *
			 *void YFromx(double *Y,
			 *            double *xnew, double *Ynew,
			 *            double lambda2, int g, int *zeroGroupFlag,
			 *            double *G, double *w)
			 */
			YFromx(YS, xnew, Ynew, lambda2, g, zeroGroupFlag, G, w);

			/*compute the solution with the starting point YS
			 *
			 *void xFromY(double *x, double *y,
			 *            double *u, double *Y,
			 *            int p, int g, int *zeroGroupFlag,
			 *            double *G, double *w)
			 *
			 */
			xFromY(xS, y, u, YS, p, g, zeroGroupFlag, G, w);

			/*compute the duality at (xS, YS)
			 *
			 * void dualityGap(double *gap, double *penalty2,
			 *               double *x, double *Y, int g, int *zeroGroupFlag,
			 *               double *G, double *w, double lambda2)
			 *
			 */
			dualityGap(&gapR, &penalty2R, xS, YS, g, zeroGroupFlag, G, w, lambda2);

			if (*gap< gapR){
				/*(xnew, Ynew) is better in terms of duality gap*/

				/*In this case, we do not apply restart, as (xS,YS) is not better
				 *
				 * We postpone the "restart" by giving a
				 *           "nextRestartStep"
				 */

				/*memcpy(xp, x, sizeof(double) * p);*/
				memcpy(Yp, Y, sizeof(double) * YSize);

				/*memcpy(x, xnew, sizeof(double) * p);*/
				memcpy(Y, Ynew, sizeof(double) * YSize);

				gamma=gamma * (1-alpha);

				alphap=alpha;

				nextRestartStep=iterStep+ (int) sqrt(gapR / *gap);
			}
			else{
				/*we use (xS, YS), as it is better in terms of duality gap*/

				*gap=gapR;
				*penalty2=penalty2R;

				if (*gap <=tol){

					memcpy(x, xS, sizeof(double) * p);
					memcpy(Y, YS, sizeof(double) * YSize);

					break;
				}else{
					/*
					 * we do a gradient descent based on  (xS, YS)
					 *
					 */

					/*
					 * compute (x, Y) from (xS, YS)
					 *
					 *
					 * gradientDescentStep(double *xnew, double *Ynew,
					 * double *LL, double *u, double *y, int *entrySignFlag, double lambda2,
					 * double *x, double *Y, int p, int g, int * zeroGroupFlag,
					 * double *G, double *w)
					 */
					gradientDescentStep(x, Y,
							&L, u, y, entrySignFlag,lambda2,
							xS, YS, p, g, zeroGroupFlag,
							G, w);

					/*memcpy(xp, xS, sizeof(double) * p);*/
					memcpy(Yp, YS, sizeof(double) * YSize);

					gamma=L;

					alphap=0.5;

				}


			}

			/*
			 * printf("\n iterStep=%d, L=%2.5f, gap=%e, gapR=%e", iterStep, L, *gap, gapR);
			 */

		}/* flag =1*/

	} /* main loop */



	penalty2[3]=iterStep+1;

	/*
	 * get the number of nonzero groups
	 */

	penalty2[4]=0;
	for(i=0;i<g;i++){
		if (zeroGroupFlag[i]==0)
			penalty2[4]=penalty2[4]+1;
		else{
			for(j=(int) w[3*i] ; j<= (int) w[3*i +1]; j++){
				if (x[ (int) G[j] ] !=0)
					break;
			}

			if (j>(int) w[3*i +1])
				penalty2[4]=penalty2[4]+1;
		}
	}


	/*
	 * assign sign to the solution x
	 */
	for(i=0;i<p;i++){
		if (entrySignFlag[i]==-1){
			x[i]=-x[i];
		}
	}

	free (u);
	free (y);

	free (xnew);
	free (Ynew);

	free (xS);
	free (YS);

	/*free (xp);*/
	free (Yp);

	free (zeroGroupFlag);
	free (entrySignFlag);
}

void overlapping(double *x, double *gap, double *penalty2,
		double *v, int p, int g, double lambda1, double lambda2,
		double *w, double *G, double *Y, int maxIter, int flag, double tol){

	switch(flag){
		case 0:
		case 1:
			overlapping_gd(x, gap, penalty2,
					v,  p, g, lambda1, lambda2,
					w, G, Y, maxIter, flag,tol);
			break;
		case 2:
		case 3:

			overlapping_agd(x, gap, penalty2,
					v,  p, g, lambda1, lambda2,
					w, G, Y, maxIter, flag-2,tol);

			break;
		default:
			/* printf("\n Wrong flag! The value of flag should be 0,1,2,3. The program uses flag=2.");*/

			overlapping_agd(x, gap, penalty2,
					v,  p, g, lambda1, lambda2,
					w, G, Y, maxIter, 0,tol);
			break;
	}


}
