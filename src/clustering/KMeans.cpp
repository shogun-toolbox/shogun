/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * 
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "clustering/KMeans.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"

#ifndef WIN32
#include <pthread.h>
#endif

#define MUSRECALC

#define PAR_THRESH  10

CKMeans::CKMeans(): max_iter(10000), k(3), dimensions(0), R(NULL), mus(NULL), Weights(NULL)
{
}

CKMeans::~CKMeans()
{
}

bool CKMeans::train()
{
	ASSERT(CDistanceMachine::get_distance());
	ASSERT(get_distance());
	ASSERT(get_distance()->get_feature_type() == F_DREAL);
	ASSERT(get_distance()->get_distance_type() == D_EUCLIDIAN);
	CRealFeatures* lhs = (CRealFeatures*) get_distance()->get_lhs();
	ASSERT(lhs);
	INT num=lhs->get_num_vectors();

	Weights=new DREAL[num];
	for (INT i=0; i<num; i++)
		Weights[i]=1.0;

	clustknb(false, NULL);
	delete[] Weights;

	return true;
}

CLabels* CKMeans::classify(CLabels* output)
{
	return NULL;
}

bool CKMeans::load(FILE* srcfile)
{
	return false;
}

bool CKMeans::save(FILE* dstfile)
{
	return false;
}

struct thread_data
{
	double* x;
	CRealFeatures* y;
	double* z;
	int n1, n2, m ; 
	int js, je ; /* defines the matrix stripe */
	int offs;
};

void *sqdist_thread_func(void * P) 
{
	struct thread_data *TD=(struct thread_data*) P;
	double* x=TD->x;
	CRealFeatures* y=TD->y;
	double* z=TD->z;
	int n1=TD->n1, 
		m=TD->m,
		js=TD->js,
		je=TD->je,
		offs=TD->offs,
		j,i,k;

	for (j=js; j<je; j++)
	{
		INT vlen=0;
		bool vfree=false;
		double* vec=y->get_feature_vector(j+offs, vlen, vfree);

		for (i=0; i<n1; i++)
		{
			double sum=0;
			for (k=0; k<m; k++) 
				sum = sum + CMath::sq(x[i*m + k] - vec[k]);
			z[j*n1 + i] = sum;
		}

		y->free_feature_vector(vec, j, vfree);
	}
	return NULL;
} 

void CKMeans::sqdist(double * x, CRealFeatures* y, double *z,
		int n1, int offs, int n2, int m)
{
	const int num_threads=parallel.get_num_threads();
	int nc, n2_nc = n2/num_threads;
	struct thread_data TD[num_threads];
	pthread_t tid[num_threads];
	void *status;

	/* prepare the structure */
	TD[0].x=x ; TD[0].y=y ; TD[0].z=z ; 
	TD[0].n1=n1 ; TD[0].n2=n2 ; TD[0].m=m;
	TD[0].offs=offs;

	if (n2>PAR_THRESH)
	{
		ASSERT(PAR_THRESH>1);

		/* create the threads */
		for (nc=0; nc<num_threads; nc++)
		{
			TD[nc]=TD[0];
			TD[nc].js=nc*n2_nc;
			if (nc+1==num_threads)
				TD[nc].je=n2;
			else
				TD[nc].je=(nc+1)*n2_nc;

			pthread_create(&tid[nc], NULL, sqdist_thread_func, (void*)&TD[nc]);
		}

		/* wait for finishing all threads */
		for (nc=0; nc<num_threads; nc++)
			pthread_join(tid[nc], &status);

	}
	else
	{
		/* simply call the ,,thread''-function */
		TD[0].js=0 ; TD[0].je=n2;
		sqdist_thread_func((void *)&TD[0]);
	}
}

void CKMeans::clustknb(bool use_old_mus, double *mus_start)
{
	ASSERT(get_distance() && get_distance()->get_feature_type() == F_DREAL);
	CRealFeatures* lhs = (CRealFeatures*) get_distance()->get_lhs();
	ASSERT(lhs && lhs->get_num_features()>0 && lhs->get_num_vectors()>0);
	
	int XSize=lhs->get_num_vectors();
	dimensions=lhs->get_num_features();
	int i, changed=1;
	const int XDimk=dimensions*k;
	int iter=0;

	delete[] R;
	R=new DREAL[k];

	delete[] mus;
	mus=new DREAL[XDimk];

	int *ClList = (int*) calloc(XSize, sizeof(int));
	double *SetWeigths = (double*) calloc(k, sizeof(double));
	double *oldmus = (double*) calloc(XDimk, sizeof(double));
	double *dists = (double*) calloc(k*XSize, sizeof(double));

	INT vlen=0;
	bool vfree=false;
	double* vec=NULL;

	/* ClList=zeros(XSize,1) ; */
	for (i=0; i<XSize; i++) ClList[i]=0;
	/* SetWeigths=zeros(k,1) ; */
	for (i=0; i<k; i++) SetWeigths[i]=0;

	/* mus=zeros(dimensions, k) ; */
	for (i=0; i<XDimk; i++) mus[i]=0;

	if (!use_old_mus)
	{
		/* random clustering (select random subsets) */
		/*  ks=ceil(rand(1,XSize)*k);
		 *  for i=1:k,
		 *	actks= (ks==i);
		 *	c=sum(actks);
		 *	SetWeigths(i)=c;
		 *
		 *	ClList(actks)=i*ones(1, c);
		 *
		 *	if ~mus_recalc,
		 *		if c>1
		 *			mus(:,i) = sum(XData(:,actks)')'/c;
		 *		elseif c>0
		 *			mus(:,i) = XData(:,actks);
		 *		end;
		 *	end;
		 *   end ; */

		for (i=0; i<XSize; i++) 
		{
			const int Cl= (int) ( CMath::random(0, k-1) );
			int j;
			double weight=Weights[i];

			SetWeigths[Cl]+=weight;
			ClList[i]=Cl;

			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
		for (i=0; i<k; i++)
		{
			int j;

			if (SetWeigths[i]!=0.0)
				for (j=0; j<dimensions; j++)
					mus[i*dimensions+j] /= SetWeigths[i];
		}
	}
	else 
	{
		ASSERT(mus_start!=NULL);

		sqdist(mus_start, lhs, dists, k, 0, XSize, dimensions);

		for (i=0; i<XSize; i++)
		{
			double mini=dists[i*k];
			int Cl = 0, j;

			for (j=1; j<k; j++)
			{
				if (dists[i*k+j]<mini)
				{
					Cl=j;
					mini=dists[i*k+j];
				}
			}
			ClList[i]=Cl;
		}

		/* Compute the sum of all points belonging to a cluster 
		 * and count the points */
		for (i=0; i<XSize; i++) 
		{
			const int Cl = ClList[i];
			double weight=Weights[i];
			SetWeigths[Cl]+=weight;
#ifndef MUSRECALC
			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
#endif
		}
#ifndef MUSRECALC
		/* normalization to get the mean */ 
		for (i=0; i<k; i++)
		{
			if (SetWeigths[i]!=0.0)
			{
				int j;
				for (j=0; j<dimensions; j++)
					mus[i*dimensions+j] /= SetWeigths[i];
			}
		}
#endif
	}

	for (i=0; i<XDimk; i++) oldmus[i]=-1;

	while (changed && (iter<max_iter))
	{
		iter++;
		if (iter==max_iter-1)
			SG_WARNING("kmeans clustering changed throughout %d iterations stopping...\n", max_iter-1);

		if (iter%1000 == 0)
			SG_INFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed);
		changed=0;

#ifdef MUSRECALC
		/* mus=zeros(dimensions, k) ; */
		for (i=0; i<XDimk; i++) mus[i]=0;

		for (i=0; i<XSize; i++) 
		{
			int j;
			int Cl=ClList[i];
			double weight=Weights[i];

			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
		for (i=0; i<k; i++)
		{
			int j;

			if (SetWeigths[i]!=0.0)
				for (j=0; j<dimensions; j++)
					mus[i*dimensions+j] /= SetWeigths[i];
		}
#endif

		for (i=0; i<XSize; i++)
		{
			/* ks=ceil(rand(1,XSize)*XSize) ; */
			const int Pat= CMath::random(0, XSize-1);
			const int ClList_Pat=ClList[Pat];
			int imini, j;
			double mini, weight;

			weight=Weights[Pat];

			/* compute the distance of this point to all centers */
			/* dists=sqdist(mus',XData) ; */
			sqdist(mus, lhs, dists, k, Pat, 1, dimensions);

			/* [mini,imini]=min(dists(:,i)) ; */
			imini=0 ; mini=dists[0];
			for (j=1; j<k; j++)
				if (dists[j]<mini)
				{
					mini=dists[j];
					imini=j;
				}

			if (imini!=ClList_Pat)
			{
				changed= changed + 1;

				/* SetWeigths(imini) = SetWeigths(imini) + weight ; */
				SetWeigths[imini]+= weight;
				/* SetWeigths(j)     = SetWeigths(j)     - weight ; */
				SetWeigths[ClList_Pat]-= weight;

				/* mu_new=mu_old + (x - mu_old)/(n+1) */
				/* mus(:,imini)=mus(:,imini) + (XData(:,i) - mus(:,imini)) * (weight / SetWeigths(imini)) ; */
				vec=lhs->get_feature_vector(Pat, vlen, vfree);

				for (j=0; j<dimensions; j++)
					mus[imini*dimensions+j]-=(vec[j]-mus[imini*dimensions+j])*(weight/SetWeigths[imini]);

				lhs->free_feature_vector(vec, Pat, vfree);

				/* mu_new = mu_old - (x - mu_old)/(n-1) */
				/* if SetWeigths(j)~=0 */
				if (SetWeigths[ClList_Pat]!=0.0)
				{
					/* mus(:,j)=mus(:,j) - (XData(:,i) - mus(:,j)) * (weight/SetWeigths(j)) ; */
					vec=lhs->get_feature_vector(Pat, vlen, vfree);

					for (j=0; j<dimensions; j++)
						mus[ClList_Pat*dimensions+j]-=(vec[j]-mus[ClList_Pat*dimensions+j])*(weight/SetWeigths[ClList_Pat]);
					lhs->free_feature_vector(vec, Pat, vfree);
				}
				else
					/*  mus(:,j)=zeros(dimensions,1) ; */
					for (j=0; j<dimensions; j++)
						mus[ClList_Pat*dimensions+j]=0;

				/* ClList(i)= imini ; */
				ClList[Pat] = imini;
			}
		}
	}

	/* compute the ,,variances'' of the clusters */
	for (i=0; i<k; i++)
	{
		double rmin1=0;
		double rmin2=0;

		bool first_round=true;

		for (INT j=0; j<k; j++) 
		{
			if (j!=i)
			{
				int l;
				double dist = 0;

				for (l=0; l<dimensions; l++)
					dist+=CMath::sq(mus[i*dimensions+l]-mus[j*dimensions+l]);

				if (first_round)
				{
					rmin1=dist;
					rmin2=dist;
					first_round=false;
				}
				else
				{
					if ((dist<rmin2) && (dist>=rmin1))
						rmin2=dist;

					if (dist<rmin1) 
					{
						rmin2=rmin1;
						rmin1=dist;
					}
				}
			}
		}

		R[i]=(0.7*sqrt(rmin1)+0.3*sqrt(rmin2));
	}

	free(ClList);
	free(SetWeigths);
	free(oldmus);
	free(dists);
} 
