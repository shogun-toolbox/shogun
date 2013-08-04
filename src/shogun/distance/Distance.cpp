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
#include <shogun/lib/Signal.h>
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

/** distance thread parameters */
template <class T> struct D_THREAD_PARAM
{
	/** distance */
	CDistance* distance;
	/** start (unit row) */
	int32_t start;
	/** end (unit row) */
	int32_t end;
	/** start (unit number of elements) */
	int32_t total_start;
	/** end (unit number of elements) */
	int32_t total_end;
	/** m */
	int32_t m;
	/** n */
	int32_t n;
	/** result */
	T* result;
	/** distance matrix k(i,j)=k(j,i) */
	bool symmetric;
	/** output progress */
	bool verbose;
};

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
	ASSERT(l)
	ASSERT(r)

	//make sure features are compatible
	ASSERT(l->get_feature_class()==r->get_feature_class())
	ASSERT(l->get_feature_type()==r->get_feature_type())

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

/// takes all necessary steps if the rhs is removed from distance
void CDistance::remove_rhs()
{
	SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;
}

CFeatures* CDistance::replace_rhs(CFeatures* r)
{
	//make sure features were indeed supplied
	ASSERT(r)

	//make sure features are compatible
	ASSERT(lhs->get_feature_class()==r->get_feature_class())
	ASSERT(lhs->get_feature_type()==r->get_feature_type())

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
	ASSERT(l)

	//make sure features are compatible
	ASSERT(rhs->get_feature_class()==l->get_feature_class())
	ASSERT(rhs->get_feature_type()==l->get_feature_type())

	//remove references to previous rhs features
	CFeatures* tmp=lhs;

	lhs=l;
	num_lhs=l->get_num_vectors();

	SG_FREE(precomputed_matrix);
	precomputed_matrix=NULL ;

	// return old features including reference count
	return tmp;
}

float64_t CDistance::distance(int32_t idx_a, int32_t idx_b)
{
	REQUIRE(idx_a >= 0 && idx_b >= 0, "In CDistance::distance(int32_t,int32_t), idx_a and "
			"idx_b must be positive, %d and %d given instead\n", idx_a, idx_b)

	ASSERT(lhs)
	ASSERT(rhs)

	if (lhs==rhs)
	{
		int32_t num_vectors = lhs->get_num_vectors();

		if (idx_a>=num_vectors)
			idx_a=2*num_vectors-1-idx_a;

		if (idx_b>=num_vectors)
			idx_b=2*num_vectors-1-idx_b;
	}

	REQUIRE(idx_a < lhs->get_num_vectors() && idx_b < rhs->get_num_vectors(),
			"In CDistance::distance(int32_t,int32_t), idx_a and idx_b must be less than "
			"the number of vectors, but %d >= %d or %d >= %d\n",
			idx_a, lhs->get_num_vectors(), idx_b, rhs->get_num_vectors())

	if (precompute_matrix && (precomputed_matrix==NULL) && (lhs==rhs))
		do_precompute_matrix() ;

	if (precompute_matrix && (precomputed_matrix!=NULL))
	{
		if (idx_a>=idx_b)
			return precomputed_matrix[idx_a*(idx_a+1)/2+idx_b] ;
		else
			return precomputed_matrix[idx_b*(idx_b+1)/2+idx_a] ;
	}

	return compute(idx_a, idx_b);
}

void CDistance::do_precompute_matrix()
{
	int32_t num_left=lhs->get_num_vectors();
	int32_t num_right=rhs->get_num_vectors();
	SG_INFO("precomputing distance matrix (%ix%i)\n", num_left, num_right) 

	ASSERT(num_left==num_right)
	ASSERT(lhs==rhs)
	int32_t num=num_left;

	SG_FREE(precomputed_matrix);
	precomputed_matrix=SG_MALLOC(float32_t, num*(num+1)/2);

	for (int32_t i=0; i<num; i++)
	{
		SG_PROGRESS(i*i,0,num*num)
		for (int32_t j=0; j<=i; j++)
			precomputed_matrix[i*(i+1)/2+j] = compute(i,j) ;
	}

	SG_PROGRESS(num*num,0,num*num)
	SG_DONE()
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

template <class T> void* CDistance::get_distance_matrix_helper(void* p)
{
	D_THREAD_PARAM<T>* params= (D_THREAD_PARAM<T>*) p;
	int32_t i_start=params->start;
	int32_t i_end=params->end;
	CDistance* k=params->distance;
	T* result=params->result;
	bool symmetric=params->symmetric;
	int32_t n=params->n;
	int32_t m=params->m;
	bool verbose=params->verbose;
	int64_t total_start=params->total_start;
	int64_t total_end=params->total_end;
	int64_t total=total_start;

	for (int32_t i=i_start; i<i_end; i++)
	{
		int32_t j_start=0;

		if (symmetric)
			j_start=i;

		for (int32_t j=j_start; j<n; j++)
		{
			float64_t v=k->distance(i,j);
			result[i+j*m]=v;

			if (symmetric && i!=j)
				result[j+i*m]=v;

			if (verbose)
			{
				total++;

				if (symmetric && i!=j)
					total++;

				if (total%100 == 0)
					SG_OBJ_PROGRESS(k, total, total_start, total_end)

				if (CSignal::cancel_computations())
					break;
			}
		}

	}

	return NULL;
}

template <class T>
SGMatrix<T> CDistance::get_distance_matrix()
{
	T* result = NULL;

	REQUIRE(has_features(), "no features assigned to distance\n")

	int32_t m=get_num_vec_lhs();
	int32_t n=get_num_vec_rhs();

	int64_t total_num = int64_t(m)*n;

	// if lhs == rhs and sizes match assume k(i,j)=k(j,i)
	bool symmetric= (lhs && lhs==rhs && m==n);

	SG_DEBUG("returning distance matrix of size %dx%d\n", m, n)

		result=SG_MALLOC(T, total_num);

	int32_t num_threads=parallel->get_num_threads();
	if (num_threads < 2)
	{
		D_THREAD_PARAM<T> params;
		params.distance=this;
		params.result=result;
		params.start=0;
		params.end=m;
		params.total_start=0;
		params.total_end=total_num;
		params.n=n;
		params.m=m;
		params.symmetric=symmetric;
		params.verbose=true;
		get_distance_matrix_helper<T>((void*) &params);
	}
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		D_THREAD_PARAM<T>* params = SG_MALLOC(D_THREAD_PARAM<T>, num_threads);
		int64_t step= total_num/num_threads;

		int32_t t;

		num_threads--;
		for (t=0; t<num_threads; t++)
		{
			params[t].distance = this;
			params[t].result = result;
			params[t].start = compute_row_start(t*step, n, symmetric);
			params[t].end = compute_row_start((t+1)*step, n, symmetric);
			params[t].total_start=t*step;
			params[t].total_end=(t+1)*step;
			params[t].n=n;
			params[t].m=m;
			params[t].symmetric=symmetric;
			params[t].verbose=false;

			int code=pthread_create(&threads[t], NULL,
					CDistance::get_distance_matrix_helper<T>, (void*)&params[t]);

			if (code != 0)
			{
				SG_WARNING("Thread creation failed (thread %d of %d) "
						"with error:'%s'\n",t, num_threads, strerror(code));
				num_threads=t;
				break;
			}
		}

		params[t].distance = this;
		params[t].result = result;
		params[t].start = compute_row_start(t*step, n, symmetric);
		params[t].end = m;
		params[t].total_start=t*step;
		params[t].total_end=total_num;
		params[t].n=n;
		params[t].m=m;
		params[t].symmetric=symmetric;
		params[t].verbose=true;
		get_distance_matrix_helper<T>(&params[t]);

		for (t=0; t<num_threads; t++)
		{
			if (pthread_join(threads[t], NULL) != 0)
				SG_WARNING("pthread_join of thread %d/%d failed\n", t, num_threads)
		}

		SG_FREE(params);
		SG_FREE(threads);
	}

	SG_DONE()

	return SGMatrix<T>(result,m,n,true);
}

template SGMatrix<float64_t> CDistance::get_distance_matrix<float64_t>();
template SGMatrix<float32_t> CDistance::get_distance_matrix<float32_t>();

template void* CDistance::get_distance_matrix_helper<float64_t>(void* p);
template void* CDistance::get_distance_matrix_helper<float32_t>(void* p);
