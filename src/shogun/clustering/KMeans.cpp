/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/clustering/KMeans.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parallel.h>

#ifndef WIN32
#include <pthread.h>
#endif

#define MUSRECALC

#define PAR_THRESH  10

using namespace shogun;

CKMeans::CKMeans()
: CDistanceMachine()
{
	init();
}

CKMeans::CKMeans(int32_t k_, CDistance* d)
: CDistanceMachine()
{
	init();
	k=k_;
	set_distance(d);
}

CKMeans::~CKMeans()
{
	R.destroy_vector();
	SG_UNREF(distance);
}

bool CKMeans::train_machine(CFeatures* data)
{
	ASSERT(distance);

	if (data)
		distance->init(data, data);

	ASSERT(distance->get_feature_type()==F_DREAL);

	CSimpleFeatures<float64_t>* lhs=
			(CSimpleFeatures<float64_t>*)distance->get_lhs();
	ASSERT(lhs);
	int32_t num=lhs->get_num_vectors();
	SG_UNREF(lhs);

	Weights=SGVector<float64_t>(num);
	for (int32_t i=0; i<num; i++)
		Weights.vector[i]=1.0;

	clustknb(false, NULL);
	Weights.destroy_vector();

	return true;
}

bool CKMeans::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CKMeans::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct thread_data
{
	float64_t* x;
	CSimpleFeatures<float64_t>* y;
	float64_t* z;
	int32_t n1, n2, m;
	int32_t js, je; /* defines the matrix stripe */
	int32_t offs;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace shogun
{
void *sqdist_thread_func(void * P)
{
	struct thread_data *TD=(struct thread_data*) P;
	float64_t* x=TD->x;
	CSimpleFeatures<float64_t>* y=TD->y;
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
}

void CKMeans::clustknb(bool use_old_mus, float64_t *mus_start)
{
	ASSERT(distance && distance->get_feature_type()==F_DREAL);
	CSimpleFeatures<float64_t>* lhs = (CSimpleFeatures<float64_t>*) distance->get_lhs();
	ASSERT(lhs && lhs->get_num_features()>0 && lhs->get_num_vectors()>0);

	int32_t XSize=lhs->get_num_vectors();
	dimensions=lhs->get_num_features();
	int32_t i, changed=1;
	const int32_t XDimk=dimensions*k;
	int32_t iter=0;

	R.destroy_vector();
	R=SGVector<float64_t>(k);

	float64_t* mus=SG_MALLOC(float64_t, XDimk);

	int32_t *ClList=SG_CALLOC(int32_t, XSize);
	float64_t *weights_set=SG_CALLOC(float64_t, k);
	float64_t *dists=SG_CALLOC(float64_t, k*XSize);

	///replace rhs feature vectors
	CSimpleFeatures<float64_t>* rhs_mus = new CSimpleFeatures<float64_t>(0);
	CFeatures* rhs_cache = distance->replace_rhs(rhs_mus);
 
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
			float64_t weight=Weights.vector[i];

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

		/// set rhs to mus_start
		rhs_mus->copy_feature_matrix(mus_start,dimensions,k);
		float64_t* p_dists=dists;

		for(int32_t idx=0;idx<XSize;idx++,p_dists+=k)
			distances_rhs(p_dists,0,k,idx);
		p_dists=NULL;            

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
			float64_t weight=Weights.vector[i];
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
			float64_t weight=Weights.vector[i];

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
		///update rhs
		rhs_mus->copy_feature_matrix(mus,dimensions,k);

		for (i=0; i<XSize; i++)
		{
			/* ks=ceil(rand(1,XSize)*XSize) ; */
			const int32_t Pat= CMath::random(0, XSize-1);
			const int32_t ClList_Pat=ClList[Pat];
			int32_t imini, j;
			float64_t mini, weight;

			weight=Weights.vector[Pat];

			/* compute the distance of this point to all centers */
			for(int32_t idx_k=0;idx_k<k;idx_k++)
				dists[idx_k]=distance->distance(Pat,idx_k);
           
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

		R.vector[i]=(0.7*CMath::sqrt(rmin1)+0.3*CMath::sqrt(rmin2));
	}
        distance->replace_rhs(rhs_cache);
        delete rhs_mus;        
	SG_FREE(ClList);
	SG_FREE(weights_set);
	SG_FREE(dists);
	SG_UNREF(lhs);

	/* set lhs of underlying distance to cluster centers */
	CSimpleFeatures<float64_t>* cluster_centers=new CSimpleFeatures<float64_t>(
			SGMatrix<float64_t>(mus, dimensions, k));
	CFeatures* rhs=distance->get_rhs();
	distance->init(cluster_centers, rhs);
}

void CKMeans::init()
{
	max_iter=10000;
	k=3;
	dimensions=0;

	m_parameters->add(&max_iter, "max_iter", "Maximum number of iterations");
	m_parameters->add(&k, "k", "Parameter k");
	m_parameters->add(&dimensions, "dimensions", "Dimensions of data");
	m_parameters->add(&R, "R", "Cluster radiuses");
}

CLabels* CKMeans::apply(CFeatures* data)
{
	/* set distance features to given ones and apply to all */
	CFeatures* lhs=distance->get_lhs();
	distance->init(lhs, data);
	SG_UNREF(lhs);

	/* build result labels and classify all elements of procedure */
	CLabels* result=new CLabels(data->get_num_vectors());
	for (index_t i=0; i<data->get_num_vectors(); ++i)
		result->set_label(i, apply(i));

	return result;
}

CLabels* CKMeans::apply()
{
	/* call apply on complete right hand side */
	CFeatures* all=distance->get_rhs();
	SG_UNREF(all);
	return apply(all);
}

float64_t CKMeans::apply(int32_t num)
{
	if (!R.vector)
		SG_ERROR("call train before calling apply!\n");

	/* number of clusters */
	CFeatures* lhs=distance->get_lhs();
	int32_t num_clusters=lhs->get_num_vectors();
	SG_UNREF(lhs);

	/* (multiple threads) calculate distances to all cluster centers */
	float64_t* dists=SG_MALLOC(float64_t, num_clusters);
	distances_lhs(dists, 0, num_clusters, num);

	/* find cluster index with smallest distance */
	float64_t result=dists[0];
	index_t best_index=0;
	for (index_t i=1; i<num_clusters; ++i)
	{
		if (dists[i]<result)
		{
			result=dists[i];
			best_index=i;
		}
	}

	SG_FREE(dists);

	return labels->get_label(best_index);
}
