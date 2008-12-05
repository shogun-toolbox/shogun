/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "clustering/KMeans.h"
#include "distance/Distance.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"

#ifndef WIN32
#include <pthread.h>
#endif

#define MUSRECALC

#define PAR_THRESH  10

CKMeans::CKMeans()
: CDistanceMachine(), max_iter(10000), k(3), dimensions(0), R(NULL),
	mus(NULL), Weights(NULL)
{
}

CKMeans::CKMeans(int32_t k_, CDistance* d)
: CDistanceMachine(), max_iter(10000), k(k_), dimensions(0), R(NULL),
	mus(NULL), Weights(NULL)
{
	set_distance(d);
}

CKMeans::~CKMeans()
{
	delete[] R;
	delete[] mus;
}

bool CKMeans::train()
{
	ASSERT(distance);
	ASSERT(distance->get_feature_type()==F_DREAL);
	ASSERT(distance->get_distance_type()==D_EUCLIDIAN);
	CRealFeatures* lhs=(CRealFeatures*) distance->get_lhs();
	ASSERT(lhs);
	int32_t num=lhs->get_num_vectors();

	Weights=new float64_t[num];
	for (int32_t i=0; i<num; i++)
		Weights[i]=1.0;

	clustknb(false, NULL);
	delete[] Weights;
	SG_UNREF(lhs);

	return true;
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
	float64_t* x;
	CRealFeatures* y;
	float64_t* z;
	int32_t n1, n2, m;
	int32_t js, je; /* defines the matrix stripe */
	int32_t offs;
};

void *sqdist_thread_func(void * P)
{
	struct thread_data *TD=(struct thread_data*) P;
	float64_t* x=TD->x;
	CRealFeatures* y=TD->y;
	float64_t* z=TD->z;
	int32_t n1=TD->n1,
		m=TD->m,
		js=TD->js,
		je=TD->je,
		offs=TD->offs,
		j,i,k;

	for (j=js; j<je; j++)
	{
		int32_t vlen=0;
		bool vfree=false;
		float64_t* vec=y->get_feature_vector(j+offs, vlen, vfree);

		for (i=0; i<n1; i++)
		{
			float64_t sum=0;
			for (k=0; k<m; k++) 
				sum = sum + CMath::sq(x[i*m + k] - vec[k]);
			z[j*n1 + i] = sum;
		}

		y->free_feature_vector(vec, j, vfree);
	}
	return NULL;
} 

void CKMeans::sqdist(
	float64_t* x, CRealFeatures* y, float64_t* z, int32_t n1, int32_t offs,
	int32_t n2, int32_t m)
{
	const int32_t num_threads=parallel.get_num_threads();
	int32_t nc, n2_nc = n2/num_threads;
	thread_data* TD = new thread_data[num_threads];
	pthread_t* tid = new pthread_t[num_threads];
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

	delete[] tid;
	delete[] TD;
}

void CKMeans::clustknb(bool use_old_mus, float64_t *mus_start)
{
	ASSERT(distance && distance->get_feature_type()==F_DREAL);
	CRealFeatures* lhs = (CRealFeatures*) distance->get_lhs();
	ASSERT(lhs && lhs->get_num_features()>0 && lhs->get_num_vectors()>0);

	int32_t XSize=lhs->get_num_vectors();
	dimensions=lhs->get_num_features();
	int32_t i, changed=1;
	const int32_t XDimk=dimensions*k;
	int32_t iter=0;

	delete[] R;
	R=new float64_t[k];

	delete[] mus;
	mus=new float64_t[XDimk];

	int32_t *ClList = (int32_t*) calloc(XSize, sizeof(int32_t));
	float64_t *weights_set = (float64_t*) calloc(k, sizeof(float64_t));
	float64_t *oldmus = (float64_t*) calloc(XDimk, sizeof(float64_t));
	float64_t *dists = (float64_t*) calloc(k*XSize, sizeof(float64_t));

	int32_t vlen=0;
	bool vfree=false;
	float64_t* vec=NULL;

	/* ClList=zeros(XSize,1) ; */
	for (i=0; i<XSize; i++) ClList[i]=0;
	/* weights_set=zeros(k,1) ; */
	for (i=0; i<k; i++) weights_set[i]=0;

	/* mus=zeros(dimensions, k) ; */
	for (i=0; i<XDimk; i++) mus[i]=0;

	if (!use_old_mus)
	{
		/* random clustering (select random subsets) */
		/*  ks=ceil(rand(1,XSize)*k);
		 *  for i=1:k,
		 *	actks= (ks==i);
		 *	c=sum(actks);
		 *	weights_set(i)=c;
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
			const int32_t Cl=CMath::random(0, k-1);
			int32_t j;
			float64_t weight=Weights[i];

			weights_set[Cl]+=weight;
			ClList[i]=Cl;

			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
		for (i=0; i<k; i++)
		{
			int32_t j;

			if (weights_set[i]!=0.0)
				for (j=0; j<dimensions; j++)
					mus[i*dimensions+j] /= weights_set[i];
		}
	}
	else 
	{
		ASSERT(mus_start);

		sqdist(mus_start, lhs, dists, k, 0, XSize, dimensions);

		for (i=0; i<XSize; i++)
		{
			float64_t mini=dists[i*k];
			int32_t Cl = 0, j;

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
			const int32_t Cl = ClList[i];
			float64_t weight=Weights[i];
			weights_set[Cl]+=weight;
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
			if (weights_set[i]!=0.0)
			{
				int32_t j;
				for (j=0; j<dimensions; j++)
					mus[i*dimensions+j] /= weights_set[i];
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
			int32_t j;
			int32_t Cl=ClList[i];
			float64_t weight=Weights[i];

			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
		for (i=0; i<k; i++)
		{
			int32_t j;

			if (weights_set[i]!=0.0)
				for (j=0; j<dimensions; j++)
					mus[i*dimensions+j] /= weights_set[i];
		}
#endif

		for (i=0; i<XSize; i++)
		{
			/* ks=ceil(rand(1,XSize)*XSize) ; */
			const int32_t Pat= CMath::random(0, XSize-1);
			const int32_t ClList_Pat=ClList[Pat];
			int32_t imini, j;
			float64_t mini, weight;

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

				/* weights_set(imini) = weights_set(imini) + weight ; */
				weights_set[imini]+= weight;
				/* weights_set(j)     = weights_set(j)     - weight ; */
				weights_set[ClList_Pat]-= weight;

				/* mu_new=mu_old + (x - mu_old)/(n+1) */
				/* mus(:,imini)=mus(:,imini) + (XData(:,i) - mus(:,imini)) * (weight / weights_set(imini)) ; */
				vec=lhs->get_feature_vector(Pat, vlen, vfree);

				for (j=0; j<dimensions; j++)
					mus[imini*dimensions+j]-=(vec[j]-mus[imini*dimensions+j])*(weight/weights_set[imini]);

				lhs->free_feature_vector(vec, Pat, vfree);

				/* mu_new = mu_old - (x - mu_old)/(n-1) */
				/* if weights_set(j)~=0 */
				if (weights_set[ClList_Pat]!=0.0)
				{
					/* mus(:,j)=mus(:,j) - (XData(:,i) - mus(:,j)) * (weight/weights_set(j)) ; */
					vec=lhs->get_feature_vector(Pat, vlen, vfree);

					for (j=0; j<dimensions; j++)
						mus[ClList_Pat*dimensions+j]-=(vec[j]-mus[ClList_Pat*dimensions+j])*(weight/weights_set[ClList_Pat]);
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
		float64_t rmin1=0;
		float64_t rmin2=0;

		bool first_round=true;

		for (int32_t j=0; j<k; j++)
		{
			if (j!=i)
			{
				int32_t l;
				float64_t dist = 0;

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
	free(weights_set);
	free(oldmus);
	free(dists);
} 
