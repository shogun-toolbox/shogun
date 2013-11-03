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
#include <shogun/labels/Labels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parallel.h>

#ifdef HAVE_PTHREAD
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

CKMeans::CKMeans(int32_t k_i, CDistance* d_i, float64_t* centers_i )
{
	init();
	k = k_i;
	set_distance(d_i);
	set_initial_centers(centers_i);
}

CKMeans::~CKMeans()
{
}

bool CKMeans::set_initial_centers(float64_t* centers)
{
	//ASSERT(len == k*get_dimensions());
	mus_initial = centers;
	mus_set = true;
	return true;	
}

bool CKMeans::train_machine(CFeatures* data)
{
	ASSERT(distance)

	if (data)
		distance->init(data, data);

	ASSERT(distance->get_feature_type()==F_DREAL)

	CDenseFeatures<float64_t>* lhs=
			(CDenseFeatures<float64_t>*)distance->get_lhs();
	ASSERT(lhs)
	int32_t num=lhs->get_num_vectors();
	SG_UNREF(lhs);

	Weights=SGVector<float64_t>(num);
	for (int32_t i=0; i<num; i++)
		Weights.vector[i]=1.0;

	if(mus_set) clustknb(true, mus_initial);
	else clustknb(false, NULL);

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


void CKMeans::set_k(int32_t p_k)
{
	ASSERT(p_k>0)
	this->k=p_k;
}

int32_t CKMeans::get_k()
{
	return k;
}

void CKMeans::set_max_iter(int32_t iter)
{
	ASSERT(iter>0)
	max_iter=iter;
}

float64_t CKMeans::get_max_iter()
{
	return max_iter;
}

SGVector<float64_t> CKMeans::get_radiuses()
{
	return R;
}

SGMatrix<float64_t> CKMeans::get_cluster_centers()
{
	if (!R.vector)
		return SGMatrix<float64_t>();

	CDenseFeatures<float64_t>* lhs=
		(CDenseFeatures<float64_t>*)distance->get_lhs();
	SGMatrix<float64_t> centers=lhs->get_feature_matrix();
	SG_UNREF(lhs);
	return centers;
}

int32_t CKMeans::get_dimensions()
{
	return dimensions;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct thread_data
{
	float64_t* x;
	CDenseFeatures<float64_t>* y;
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
	CDenseFeatures<float64_t>* y=TD->y;
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
	ASSERT(distance && distance->get_feature_type()==F_DREAL)
	CDenseFeatures<float64_t>* lhs = (CDenseFeatures<float64_t>*) distance->get_lhs();
	ASSERT(lhs && lhs->get_num_features()>0 && lhs->get_num_vectors()>0)

	int32_t XSize=lhs->get_num_vectors();
	dimensions=lhs->get_num_features();
	int32_t i, changed=1;
	const int32_t XDimk=dimensions*k;
	int32_t iter=0;

	R=SGVector<float64_t>(k);

	mus=SGMatrix<float64_t>(dimensions, k);

	int32_t *ClList=SG_CALLOC(int32_t, XSize);
	float64_t *weights_set=SG_CALLOC(float64_t, k);
	float64_t *dists=SG_CALLOC(float64_t, k*XSize);

	///replace rhs feature vectors
	CDenseFeatures<float64_t>* rhs_mus = new CDenseFeatures<float64_t>(0);
	CFeatures* rhs_cache = distance->replace_rhs(rhs_mus);

	int32_t vlen=0;
	bool vfree=false;
	float64_t* vec=NULL;

	/* ClList=zeros(XSize,1) ; */
	memset(ClList, 0, sizeof(int32_t)*XSize);
	/* weights_set=zeros(k,1) ; */
	memset(weights_set, 0, sizeof(float64_t)*k);

	/* cluster_centers=zeros(dimensions, k) ; */
	memset(mus.matrix, 0, sizeof(float64_t)*XDimk);

	if (!use_old_mus)
	{
		for (i=0; i<XSize; i++)
		{
			const int32_t Cl=CMath::random(0, k-1);
			int32_t j;
			float64_t weight=Weights.vector[i];

			weights_set[Cl]+=weight;
			ClList[i]=Cl;

			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus.matrix[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
		for (i=0; i<k; i++)
		{
			int32_t j;

			if (weights_set[i]!=0.0)
				for (j=0; j<dimensions; j++)
					mus.matrix[i*dimensions+j] /= weights_set[i];
		}
	}
	else
	{
		ASSERT(mus_start)

		/// set rhs to mus_start
		rhs_mus->copy_feature_matrix(SGMatrix<float64_t>(mus_start,dimensions,k,false));
		for(int32_t idx=0;idx<XSize;idx++)
			for(int32_t m=0;m<k;m++)
				dists[k*idx+m] = distance->distance(idx,m);

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
				mus.matrix[Cl*dimensions+j] += weight*vec[j];

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
					mus.matrix[i*dimensions+j] /= weights_set[i];
			}
		}
#endif
	}



	while (changed && (iter<max_iter))
	{
		iter++;
		if (iter==max_iter-1)
			SG_WARNING("kmeans clustering changed throughout %d iterations stopping...\n", max_iter-1)

		if (iter%1000 == 0)
			SG_INFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed)
		changed=0;

#ifdef MUSRECALC
		/* mus=zeros(dimensions, k) ; */
		memset(mus.matrix, 0, sizeof(float64_t)*XDimk);

		for (i=0; i<XSize; i++)
		{
			int32_t j;
			int32_t Cl=ClList[i];
			float64_t weight=Weights.vector[i];

			vec=lhs->get_feature_vector(i, vlen, vfree);

			for (j=0; j<dimensions; j++)
				mus.matrix[Cl*dimensions+j] += weight*vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
		for (i=0; i<k; i++)
		{
			int32_t j;

			if (weights_set[i]!=0.0)
				for (j=0; j<dimensions; j++)
					mus.matrix[i*dimensions+j] /= weights_set[i];
		}
#endif
		///update rhs
		rhs_mus->copy_feature_matrix(mus);

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

				vec=lhs->get_feature_vector(Pat, vlen, vfree);

				for (j=0; j<dimensions; j++)
				{
					mus.matrix[imini*dimensions+j]-=(vec[j]
							-mus.matrix[imini*dimensions+j])
							*(weight/weights_set[imini]);
				}

				lhs->free_feature_vector(vec, Pat, vfree);

				/* mu_new = mu_old - (x - mu_old)/(n-1) */
				/* if weights_set(j)~=0 */
				if (weights_set[ClList_Pat]!=0.0)
				{
					vec=lhs->get_feature_vector(Pat, vlen, vfree);

					for (j=0; j<dimensions; j++)
					{
						mus.matrix[ClList_Pat*dimensions+j]-=
								(vec[j]
										-mus.matrix[ClList_Pat
												*dimensions+j])
										*(weight/weights_set[ClList_Pat]);
					}
					lhs->free_feature_vector(vec, Pat, vfree);
				}
				else
					/*  mus(:,j)=zeros(dimensions,1) ; */
					for (j=0; j<dimensions; j++)
						mus.matrix[ClList_Pat*dimensions+j]=0;

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
				{
					dist+=CMath::sq(
							mus.matrix[i*dimensions+l]
									-mus.matrix[j*dimensions+l]);
				}

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
}

void CKMeans::store_model_features()
{
	/* set lhs of underlying distance to cluster centers */
	CDenseFeatures<float64_t>* cluster_centers=new CDenseFeatures<float64_t>(
			mus);

	/* store cluster centers in lhs of distance variable */
	CFeatures* rhs=distance->get_rhs();
	distance->init(cluster_centers, rhs);
	SG_UNREF(rhs);
}

void CKMeans::init()
{
	max_iter=10000;
	k=3;
	dimensions=0;

	SG_ADD(&max_iter, "max_iter", "Maximum number of iterations", MS_AVAILABLE);
	SG_ADD(&k, "k", "k, the number of clusters", MS_AVAILABLE);
	SG_ADD(&dimensions, "dimensions", "Dimensions of data", MS_NOT_AVAILABLE);
	SG_ADD(&R, "R", "Cluster radiuses", MS_NOT_AVAILABLE);
}

