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

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct DISTANCE_THREAD_PARAM
{
	// CDistance instance used by thread to compute distance
	CDistance* distance;
	// distance matrix to store computed distances
	float64_t* distance_matrix;
	// starting index of the main loop
	int32_t idx_start;
	// end index of the main loop
	int32_t idx_stop;
	// step of the main loop
	int32_t idx_step;
	// number of lhs vectors
	int32_t lhs_vectors_number;
	// number of rhs vectors
	int32_t rhs_vectors_number;
	// whether matrix distance is symmetric
	bool symmetric;
	// chunking method
	bool chunk_by_lhs;
};
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

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

	SG_UNREF(lhs);
	lhs = NULL;
}

void CDistance::remove_lhs()
{ 
	SG_UNREF(lhs);
	lhs = NULL;
}

/// takes all necessary steps if the rhs is removed from kernel
void CDistance::remove_rhs()
{
	SG_UNREF(rhs);
	rhs = NULL;
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

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()))
			SG_ERROR("distance matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		int64_t total_num=num_vec1*num_vec2;
		int32_t num_done=0;

		SG_DEBUG("returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=SG_MALLOC(float32_t, total_num);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
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
	if (!lhs_features || !rhs_features)
		SG_ERROR("No features assigned to the distance.\n");

	if (target && 
	    (lhs_vectors_number!=lhs_features->get_num_vectors() ||
	     rhs_vectors_number!=rhs_features->get_num_vectors()))
		SG_ERROR("Distance matrix does not fit into the given target.\n");

	// init numbers of vectors and total number of distances
	lhs_vectors_number = lhs_features->get_num_vectors();
	rhs_vectors_number = rhs_features->get_num_vectors();
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

#ifdef HAVE_PTHREAD
	// init parallel to work
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	DISTANCE_THREAD_PARAM* parameters = SG_MALLOC(DISTANCE_THREAD_PARAM,num_threads);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	// run threads
	for (int32_t t=0; t<num_threads; t++)
	{
		parameters[t].idx_start = t;
		parameters[t].idx_stop = chunk_by_lhs ? lhs_vectors_number : rhs_vectors_number;
		parameters[t].idx_step = num_threads;
		parameters[t].distance_matrix = distance_matrix;
		parameters[t].symmetric = symmetric;
		parameters[t].lhs_vectors_number = lhs_vectors_number;
		parameters[t].rhs_vectors_number = rhs_vectors_number;
		parameters[t].chunk_by_lhs = chunk_by_lhs;
		parameters[t].distance = this;
		pthread_create(&threads[t], &attr, run_distance_thread, (void*)&parameters[t]);
	}
	// join, i.e. wait threads for finish
	for (int32_t t=0; t<num_threads; t++)
	{
		pthread_join(threads[t], NULL);
	}
	// cleanup
	pthread_attr_destroy(&attr);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	// init one-threaded parameters
	DISTANCE_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_stop = chunk_by_lhs ? lhs_vectors_number : rhs_vectors_number;
	single_thread_param.idx_step = 1;
	single_thread_param.distance_matrix = distance_matrix;
	single_thread_param.symmetric = symmetric;
	single_thread_param.lhs_vectors_number = lhs_vectors_number;
	single_thread_param.rhs_vectors_number = rhs_vectors_number;
	single_thread_param.chunk_by_lhs = chunk_by_lhs;
	single_thread_param.distance = this;
	// run thread
	run_distance_thread((void*)&single_thread_param);
#endif

	return distance_matrix;
}

void CDistance::init()
{
	precomputed_matrix = NULL;
	precompute_matrix = false;
	lhs = NULL;
	rhs = NULL;

	m_parameters->add((CSGObject**) &lhs, "lhs",
					  "Feature vectors to occur on left hand side.");
	m_parameters->add((CSGObject**) &rhs, "rhs",
					  "Feature vectors to occur on right hand side.");
}

void* CDistance::run_distance_thread(void* p)
{
	DISTANCE_THREAD_PARAM* parameters = (DISTANCE_THREAD_PARAM*)p;
	float64_t* distance_matrix = parameters->distance_matrix;
	CDistance* distance = parameters->distance;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_stop = parameters->idx_stop;
	int32_t idx_step = parameters->idx_step;
	int32_t lhs_vectors_number = parameters->lhs_vectors_number;
	int32_t rhs_vectors_number = parameters->rhs_vectors_number;
	bool symmetric = parameters->symmetric;
	bool chunk_by_lhs = parameters->chunk_by_lhs;

	if (symmetric)
	{
		for (int32_t i=idx_start; i<idx_stop; i+=idx_step)
		{
			for (int32_t j=i; j<rhs_vectors_number; j++)
			{
				float64_t ij_distance = distance->compute(i,j);
				distance_matrix[i*rhs_vectors_number+j] = ij_distance;
				distance_matrix[j*rhs_vectors_number+i] = ij_distance;
			}
		}
	}
	else
	{
		if (chunk_by_lhs)
		{
			for (int32_t i=idx_start; i<idx_stop; i+=idx_step)
			{
				for (int32_t j=0; j<rhs_vectors_number; j++)
				{
					distance_matrix[j*lhs_vectors_number+i] = distance->compute(i,j);
				}
			}
		}
		else
		{
			for (int32_t j=idx_start; j<idx_stop; j+=idx_step)
			{
				for (int32_t i=0; i<lhs_vectors_number; i++)
				{
					distance_matrix[j*lhs_vectors_number+i] = distance->compute(i,j);
				}
			}
		}
	}

	return NULL;
}
