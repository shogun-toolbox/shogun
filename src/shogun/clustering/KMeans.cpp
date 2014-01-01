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

CKMeans::CKMeans(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i)
: CDistanceMachine()
{
	init();
	k = k_i;
	set_distance(d_i);
	set_initial_centers(centers_i);
}

CKMeans::~CKMeans()
{
}

void CKMeans::set_initial_centers(SGMatrix<float64_t> centers)
{
	dimensions = ((CDenseFeatures<float64_t>*) distance->get_lhs())->get_num_features();
	REQUIRE(centers.num_cols == k,
			"Expected %d initial cluster centers, got %d", k, centers.num_cols);
	REQUIRE(centers.num_rows == dimensions,
			"Expected %d dimensionional cluster centers, got %d", dimensions, centers.num_rows);
	mus_initial = centers;
}

void CKMeans::set_random_centers(float64_t* weights_set, int32_t* ClList, int32_t XSize)
{
	CDenseFeatures<float64_t>* lhs=
			(CDenseFeatures<float64_t>*)distance->get_lhs();

	for (int32_t i=0; i<XSize; i++)
	{
		const int32_t Cl=CMath::random(0, k-1);
		weights_set[Cl]+=1.0;
		ClList[i]=Cl;

		int32_t vlen=0;
		bool vfree=false;
		float64_t* vec=lhs->get_feature_vector(i, vlen, vfree);

		for (int32_t j=0; j<dimensions; j++)
			mus.matrix[Cl*dimensions+j] += vec[j];

		lhs->free_feature_vector(vec, i, vfree);
	}

	SG_UNREF(lhs);

	for (int32_t i=0; i<k; i++)
	{
		if (weights_set[i]!=0.0)
		{
			for (int32_t j=0; j<dimensions; j++)
				mus.matrix[i*dimensions+j] /= weights_set[i];
		}
	}
}

void CKMeans::set_initial_centers(CDenseFeatures<float64_t>* rhs_mus, float64_t* weights_set,
				float64_t* dists, int32_t* ClList, int32_t XSize)
{
	ASSERT(mus_initial.matrix);

	/// set rhs to mus_start
	rhs_mus->copy_feature_matrix(mus_initial);

	for(int32_t idx=0;idx<XSize;idx++)
	{
		for(int32_t m=0;m<k;m++)
			dists[k*idx+m] = distance->distance(idx,m);
	}

	for (int32_t i=0; i<XSize; i++)
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
	CDenseFeatures<float64_t>* lhs=
			(CDenseFeatures<float64_t>*)distance->get_lhs();
	for (int32_t i=0; i<XSize; i++)
	{
		const int32_t Cl = ClList[i];
		weights_set[Cl]+=1.0;
		if (fixed_centers)
		{
			int32_t vlen=0;
			bool vfree=false;
			float64_t* vec=lhs->get_feature_vector(i, vlen, vfree);

			for (int32_t j=0; j<dimensions; j++)
				mus.matrix[Cl*dimensions+j] += vec[j];

			lhs->free_feature_vector(vec, i, vfree);
		}
	}
	SG_UNREF(lhs);

	if (fixed_centers)
	{
		/* normalization to get the mean */
		for (int32_t i=0; i<k; i++)
		{
			if (weights_set[i]!=0.0)
			{
				for (int32_t j=0; j<dimensions; j++)
					mus.matrix[i*dimensions+j] /= weights_set[i];
			}
		}
	}
}

void CKMeans::compute_cluster_variances()
{
	/* compute the ,,variances'' of the clusters */
	for (int32_t i=0; i<k; i++)
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
}

bool CKMeans::train_machine(CFeatures* data)
{
	ASSERT(distance && distance->get_feature_type()==F_DREAL)

	if (data)
		distance->init(data, data);

	CDenseFeatures<float64_t>* lhs=
			(CDenseFeatures<float64_t>*)distance->get_lhs();

	ASSERT(lhs);
	int32_t XSize=lhs->get_num_vectors();
	dimensions=lhs->get_num_features();

	ASSERT(XSize>0 && dimensions>0);

	int32_t changed=1;
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

	if (mus_initial.matrix)
		set_initial_centers(rhs_mus, weights_set, dists, ClList, XSize);
	else
		set_random_centers(weights_set, ClList, XSize);

	while (changed && (iter<max_iter))
	{
		iter++;
		if (iter==max_iter-1)
			SG_WARNING("kmeans clustering changed throughout %d iterations stopping...\n", max_iter-1)

		if (iter%1000 == 0)
			SG_INFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed)
		changed=0;

		if (!fixed_centers)
		{
			/* mus=zeros(dimensions, k) ; */
			memset(mus.matrix, 0, sizeof(float64_t)*XDimk);

			for (int32_t i=0; i<XSize; i++)
			{
				int32_t Cl=ClList[i];

				vec=lhs->get_feature_vector(i, vlen, vfree);

				for (int32_t j=0; j<dimensions; j++)
					mus.matrix[Cl*dimensions+j] += vec[j];

				lhs->free_feature_vector(vec, i, vfree);
			}

			for (int32_t i=0; i<k; i++)
			{
				if (weights_set[i]!=0.0)
				{
					for (int32_t j=0; j<dimensions; j++)
						mus.matrix[i*dimensions+j] /= weights_set[i];
				}
			}
		}
		///update rhs
		rhs_mus->copy_feature_matrix(mus);

		for (int32_t i=0; i<XSize; i++)
		{
			/* ks=ceil(rand(1,XSize)*XSize) ; */
			const int32_t Pat= CMath::random(0, XSize-1);
			const int32_t ClList_Pat=ClList[Pat];
			int32_t imini, j;
			float64_t mini;

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
				changed++;

				/* weights_set(imini) = weights_set(imini) + 1.0 ; */
				weights_set[imini]+= 1.0;
				/* weights_set(j)     = weights_set(j)     - 1.0 ; */
				weights_set[ClList_Pat]-= 1.0;

				vec=lhs->get_feature_vector(Pat, vlen, vfree);

				for (j=0; j<dimensions; j++)
				{
					mus.matrix[imini*dimensions+j]-=
						(vec[j]-mus.matrix[imini*dimensions+j]) / weights_set[imini];
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
								(vec[j]-mus.matrix[ClList_Pat*dimensions+j]) / weights_set[ClList_Pat];
					}
					lhs->free_feature_vector(vec, Pat, vfree);
				}
				else
				{
					/*  mus(:,j)=zeros(dimensions,1) ; */
					for (j=0; j<dimensions; j++)
						mus.matrix[ClList_Pat*dimensions+j]=0;
				}

				/* ClList(i)= imini ; */
				ClList[Pat] = imini;
			}
		}
	}

	compute_cluster_variances();
	distance->replace_rhs(rhs_cache);
	delete rhs_mus;
	SG_FREE(ClList);
	SG_FREE(weights_set);
	SG_FREE(dists);
	SG_UNREF(lhs);

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

void CKMeans::set_fixed_centers(bool fixed)
{
	fixed_centers=fixed;
}

bool CKMeans::get_fixed_centers()
{
	return fixed_centers;
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
	fixed_centers=false;

	SG_ADD(&max_iter, "max_iter", "Maximum number of iterations", MS_AVAILABLE);
	SG_ADD(&k, "k", "k, the number of clusters", MS_AVAILABLE);
	SG_ADD(&dimensions, "dimensions", "Dimensions of data", MS_NOT_AVAILABLE);
	SG_ADD(&R, "R", "Cluster radiuses", MS_NOT_AVAILABLE);
}

