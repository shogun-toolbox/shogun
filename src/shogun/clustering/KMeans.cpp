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

CKMeans::CKMeans(int32_t f)
:CDistanceMachine()
{
	init();
	use_fast=f;
}

CKMeans::CKMeans(int32_t k_, CDistance* d, int32_t f)
:CDistanceMachine()
{
	init();
	k=k_;
	set_distance(d);
	use_fast=f;
}

CKMeans::CKMeans(int32_t k_, CDistance* d, bool use_kmpp, int32_t f)
: CDistanceMachine()
{	
	init();
	k=k_;
	set_distance(d);
	use_kmeanspp=use_kmpp;
	use_fast=f;
}

CKMeans::CKMeans(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i, int32_t f)
: CDistanceMachine()
{
	init();
	k = k_i;
	set_distance(d_i);
	set_initial_centers(centers_i);
	use_fast=f;
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

SGVector<int32_t> CKMeans::mbchoose_rand(int32_t b, int32_t num)
{
	SGVector<int32_t> chosen=SGVector<int32_t>(num);
	SGVector<int32_t> ret=SGVector<int32_t>(b);
	chosen.zero();
	int32_t ch=0;
	while (ch<b)
	{
		const int32_t n=CMath::random(0,num-1);
		if (chosen[n]==0)
		{
			chosen[n]+=1;
			ret[ch]=n;
			ch++;
		}
	}
	return ret;
}

void CKMeans::mbKMeans()
{
	REQUIRE(batch_size>0,
		"batch size not set to positive value. Current batch size %d \n", batch_size);
	REQUIRE(minib_iter>0,
		"number of iterations not set to positive value. Current iterations %d \n", minib_iter);

	CDenseFeatures<float64_t>* lhs=(CDenseFeatures<float64_t>*) distance->get_lhs();
	CDenseFeatures<float64_t>* rhs_mus=(CDenseFeatures<float64_t>*) distance->get_rhs();
	int32_t XSize=lhs->get_num_vectors();
	int32_t dims=lhs->get_num_features();

	SGVector<float64_t> v=SGVector<float64_t>(k);
	v.zero();

	for (int32_t i=0; i<minib_iter; i++)
	{
		SGVector<int32_t> M=mbchoose_rand(batch_size,XSize);
		SGVector<int32_t> ncent= SGVector<int32_t>(batch_size);
		for (int32_t j=0; j<batch_size; j++)
		{
			SGVector<float64_t> dists=SGVector<float64_t>(k);
			for (int32_t p=0; p<k; p++)
				dists[p]=distance->distance(M[j],p);

			int32_t imin=0;
			float64_t min=dists[0];
			for (int32_t p=1; p<k; p++)
			{
				if (dists[p]<min)
				{
					imin=p;
					min=dists[p];
				}
			}
			ncent[j]=imin;
		}
		for (int32_t j=0; j<batch_size; j++)
		{
			int32_t near=ncent[j];
			SGVector<float64_t> c_alive=rhs_mus->get_feature_vector(near);
			SGVector<float64_t> x=lhs->get_feature_vector(M[j]);
			v[near]+=1.0;
			float64_t eta=1.0/v[near];
			for (int32_t c=0; c<dims; c++)
			{
				c_alive[c]=(1.0-eta)*c_alive[c]+eta*x[c];
			}
		}
	}
	SG_UNREF(lhs);
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

	///if kmeans++ to be used
	if (use_kmeanspp)
		mus_initial=kmeanspp();

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
	
	if (use_fast==1)
	{
		rhs_mus->set_feature_matrix(mus);
		mbKMeans();
	}
	else
	{

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

void CKMeans::set_use_kmeanspp(bool kmpp)
{
	use_kmeanspp=kmpp;
}

bool CKMeans::get_use_kmeanspp() const
{
	return use_kmeanspp;
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

void CKMeans::set_use_fast(int32_t f)
{
	use_fast=f;
}

int32_t CKMeans::get_use_fast()
{
	return use_fast;
}

void CKMeans::set_mbKMeans_batch_size(int32_t b)
{
	ASSERT(b>0)
	batch_size=b;
}

int32_t CKMeans::get_mbKMeans_batch_size()
{
	return batch_size;
}

void CKMeans::set_mbKMeans_iter(int32_t i)
{
	ASSERT(i>0)
	minib_iter=i;
}

int32_t CKMeans::get_mbKMeans_iter()
{
	return minib_iter;
}

void CKMeans::set_mbKMeans_params(int32_t b, int32_t t)
{
	ASSERT(b>0)
	ASSERT(t>0)
	batch_size=b;
	minib_iter=t;
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

SGMatrix<float64_t> CKMeans::kmeanspp()
{
	int32_t num=distance->get_num_vec_lhs();
	SGVector<float64_t> dists=SGVector<float64_t>(num);
	SGVector<int32_t> mu_index=SGVector<int32_t>(k);
	
	/* 1st center */
	int32_t mu_1=CMath::random((int32_t) 0,num-1);
	mu_index[0]=mu_1;
	
	/* choose a center - do k-1 times */
	int32_t count=0;
	while (++count<k)
	{
		float64_t sum=0.0;
		/* for each data point find distance to nearest already chosen center */
		for (int32_t point_idx=0;point_idx<num;point_idx++)
		{
			dists[point_idx]=distance->distance(mu_index[0],point_idx);
			int32_t cent_id=1;

			while (cent_id<count)
			{
				float64_t dist_temp=distance->distance(mu_index[cent_id],point_idx); 
				if (dists[point_idx]>dist_temp)
					dists[point_idx]=dist_temp; 
				cent_id++;
			}

			dists[point_idx]*=dists[point_idx];
			sum+=dists[point_idx];
		}

		/*random choosing - points weighted by square of distance from nearset center*/
		int32_t mu_next=0;
		float64_t chosen=CMath::random(0.0,sum);
		while ((chosen-=dists[mu_next])>0)
			mu_next++;

		mu_index[count]=mu_next;
	}

	CDenseFeatures<float64_t>* lhs=(CDenseFeatures<float64_t>*)distance->get_lhs();
	int32_t dim=lhs->get_num_features();
	SGMatrix<float64_t> mat=SGMatrix<float64_t>(dim,k);
	for (int32_t c_m=0;c_m<k;c_m++)
	{
		SGVector<float64_t> feature=lhs->get_feature_vector(c_m);
		for (int32_t r_m=0;r_m<dim;r_m++)
			mat(r_m,c_m)=feature[r_m];
	}
	SG_UNREF(lhs);
	return mat;
}

void CKMeans::init()
{
	max_iter=10000;
	k=3;
	dimensions=0;
	fixed_centers=false;
	use_kmeanspp=false;
	use_fast=0;
	batch_size=-1;
	minib_iter=-1;
	SG_ADD(&max_iter, "max_iter", "Maximum number of iterations", MS_AVAILABLE);
	SG_ADD(&k, "k", "k, the number of clusters", MS_AVAILABLE);
	SG_ADD(&dimensions, "dimensions", "Dimensions of data", MS_NOT_AVAILABLE);
	SG_ADD(&R, "R", "Cluster radiuses", MS_NOT_AVAILABLE);
}

