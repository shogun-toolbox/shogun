/*
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>
#include <shogun/lib/Time.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Parameter.h>

#include <shogun/distance/Distance.h>
#include <shogun/features/Features.h>

#include <string.h>
#include <unistd.h>
#include <omp.h>

using namespace shogun;

CDistance::CDistance() : CSGObject()
{
	init();
}

CDistance::CDistance(CFeatures* p_lhs, CFeatures* p_rhs) : CSGObject()
{
	init();
	init(p_lhs, p_rhs);
}

CDistance::~CDistance()
{
	SG_FREE(precomputed_matrix);
	precomputed_matrix=NULL;

	remove_lhs_and_rhs();
}

bool CDistance::init(CFeatures* l, CFeatures* r)
{
	//make sure features were indeed supplied
	ASSERT(l);
	ASSERT(r);

	//make sure features are compatible
	ASSERT(l->get_feature_class()==r->get_feature_class());
	ASSERT(l->get_feature_type()==r->get_feature_type());

	//remove references to previous features
	remove_lhs_and_rhs();

	//increase reference counts
	SG_REF(l);
	SG_REF(r);

	lhs=l;
	rhs=r;

	num_lhs=l->get_num_vectors();
	num_rhs=r->get_num_vectors();

	SG_FREE(precomputed_matrix);
	precomputed_matrix=NULL ;

	return true;
}

void CDistance::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void CDistance::save(CFile* writer)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void CDistance::remove_lhs_and_rhs()
{
	SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;

	SG_UNREF(lhs);
	lhs = NULL;
	num_lhs=0;
}

void CDistance::remove_lhs()
{
	SG_UNREF(lhs);
	lhs = NULL;
	num_lhs=0;
}

/// takes all necessary steps if the rhs is removed from kernel
void CDistance::remove_rhs()
{
	SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;
}

CFeatures* CDistance::replace_rhs(CFeatures* r)
{
	//make sure features were indeed supplied
	ASSERT(r);

	//make sure features are compatible
	ASSERT(lhs->get_feature_class()==r->get_feature_class());
	ASSERT(lhs->get_feature_type()==r->get_feature_type());

	//remove references to previous rhs features
	CFeatures* tmp=rhs;

	rhs=r;
	num_rhs=r->get_num_vectors();

	SG_FREE(precomputed_matrix);
	precomputed_matrix=NULL ;

	// return old features including reference count
	return tmp;
}

CFeatures* CDistance::replace_lhs(CFeatures* l)
{
	//make sure features were indeed supplied
	ASSERT(l);

	//make sure features are compatible
	ASSERT(rhs->get_feature_class()==l->get_feature_class());
	ASSERT(rhs->get_feature_type()==l->get_feature_type());

	//remove references to previous rhs features
	CFeatures* tmp=lhs;

	lhs=l;
	num_lhs=l->get_num_vectors();

	SG_FREE(precomputed_matrix);
	precomputed_matrix=NULL ;

	// return old features including reference count
	return tmp;
}

void CDistance::do_precompute_matrix()
{
	int32_t num_left=lhs->get_num_vectors();
	int32_t num_right=rhs->get_num_vectors();
	SG_INFO( "precomputing distance matrix (%ix%i)\n", num_left, num_right) ;

	ASSERT(num_left==num_right);
	ASSERT(lhs==rhs);
	int32_t num=num_left;

	SG_FREE(precomputed_matrix);
	precomputed_matrix=SG_MALLOC(float32_t, num*(num+1)/2);

	for (int32_t i=0; i<num; i++)
	{
		SG_PROGRESS(i*i,0,num*num);
		for (int32_t j=0; j<=i; j++)
			precomputed_matrix[i*(i+1)/2+j] = compute(i,j) ;
	}

	SG_PROGRESS(num*num,0,num*num);
	SG_DONE();
}

SGMatrix<float64_t> CDistance::get_distance_matrix()
{
	int32_t m,n;
	float64_t* data=get_distance_matrix_real(m,n,NULL);
	return SGMatrix<float64_t>(data, m,n);
}

float32_t* CDistance::get_distance_matrix_shortreal(
	int32_t &num_vec1, int32_t &num_vec2, float32_t* target)
{
	float32_t* result = NULL;
	CFeatures* f1 = lhs;
	CFeatures* f2 = rhs;

	if (has_features())
	{
		if (target && (num_vec1!=get_num_vec_lhs() || num_vec2!=get_num_vec_rhs()))
			SG_ERROR("distance matrix does not fit into target\n");

		num_vec1=get_num_vec_lhs();
		num_vec2=get_num_vec_rhs();
		int64_t total_num=num_vec1*num_vec2;
		int32_t num_done=0;

		SG_DEBUG("returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=SG_MALLOC(float32_t, total_num);

		if ( (f1 == f2) && (num_vec1 == num_vec2) && (f1!=NULL && f2!=NULL) )
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=distance(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      		SG_ERROR("no features assigned to distance\n");

	return result;
}

float64_t* CDistance::get_distance_matrix_real(
	int32_t &lhs_vectors_number, int32_t &rhs_vectors_number, float64_t* target)
{
	float64_t* distance_matrix = NULL;
	CFeatures* lhs_features = lhs;
	CFeatures* rhs_features = rhs;

	// check for errors
	if (!has_features())
		SG_ERROR("No features assigned to the distance.\n");

	if (target &&
	    (lhs_vectors_number!=get_num_vec_lhs() ||
	     rhs_vectors_number!=get_num_vec_rhs()))
		SG_ERROR("Distance matrix does not fit into the given target.\n");

	// init numbers of vectors and total number of distances
	lhs_vectors_number = get_num_vec_lhs();
	rhs_vectors_number = get_num_vec_rhs();
	int64_t total_distances_number = lhs_vectors_number*rhs_vectors_number;

	SG_DEBUG("Calculating distance matrix of size %dx%d.\n", lhs_vectors_number, rhs_vectors_number);

	// redirect to target or allocate memory
	if (target)
		distance_matrix = target;
	else
		distance_matrix = SG_MALLOC(float64_t, total_distances_number);

	// check if we're computing symmetric distance_matrix
	bool symmetric = (lhs_features==rhs_features) || (lhs_vectors_number==rhs_vectors_number);
	// select chunking method according to greatest dimension
	bool chunk_by_lhs = (lhs_vectors_number >= rhs_vectors_number);

	int32_t i,j;
	if (symmetric)
	{
#pragma omp parallel private(i,j)
#pragma omp for schedule(dynamic) nowait
		for (i=0; i<lhs_vectors_number; i++)
		{
			for (j=i; j<rhs_vectors_number; j++)
			{
				float64_t ij_distance = compute(i,j);
				distance_matrix[i*rhs_vectors_number+j] = ij_distance;
				distance_matrix[j*rhs_vectors_number+i] = ij_distance;
			}
		}
	}
	else
	{
		if (chunk_by_lhs)
		{
#pragma omp parallel private(i,j)
#pragma omp for schedule(dynamic) nowait
			for (i=0; i<lhs_vectors_number; i++)
			{
				for (j=0; j<rhs_vectors_number; j++)
				{
					distance_matrix[j*lhs_vectors_number+i] = compute(i,j);
				}
			}
		}
		else
		{
#pragma omp parallel private(i,j)
#pragma omp for schedule(dynamic) nowait
			for (j=0; j<rhs_vectors_number; j++)
			{
				for (i=0; i<lhs_vectors_number; i++)
				{
					distance_matrix[j*lhs_vectors_number+i] = compute(i,j);
				}
			}
		}
	}
	return distance_matrix;
}

void CDistance::init()
{
	precomputed_matrix = NULL;
	precompute_matrix = false;
	lhs = NULL;
	rhs = NULL;
	num_lhs=0;
	num_rhs=0;

	m_parameters->add((CSGObject**) &lhs, "lhs",
					  "Feature vectors to occur on left hand side.");
	m_parameters->add((CSGObject**) &rhs, "rhs",
					  "Feature vectors to occur on right hand side.");
}
