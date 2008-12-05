/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "base/Parallel.h"

#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/CustomKernel.h"
#include "features/CombinedFeatures.h"

#include <string.h>

#ifndef WIN32
#include <pthread.h>
#endif

struct S_THREAD_PARAM
{
	CKernel* kernel;
	float64_t* result;
	int32_t* vec_idx;
	int32_t start;
	int32_t end;
	/// required only for non optimized kernels
	float64_t* weights;
	int32_t* IDX;
	int32_t num_suppvec;
};

CCombinedKernel::CCombinedKernel(int32_t size, bool asw)
: CKernel(size), sv_count(0), sv_idx(NULL), sv_weight(NULL),
	subkernel_weights_buffer(NULL), append_subkernel_weights(asw)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	kernel_list=new CList<CKernel*>(true);
	SG_INFO("Combined kernel created (%p)\n", this) ;
	if (append_subkernel_weights)
		SG_INFO( "(subkernel weights are appended)\n") ;
}

CCombinedKernel::CCombinedKernel(
	CCombinedFeatures *l, CCombinedFeatures *r, bool asw)
: CKernel(10), sv_count(0), sv_idx(NULL), sv_weight(NULL),
	subkernel_weights_buffer(NULL), append_subkernel_weights(asw)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	kernel_list=new CList<CKernel*>(true);
	SG_INFO("Combined kernel created (%p)\n", this) ;
	if (append_subkernel_weights) {
		SG_INFO("(subkernel weights are appended)\n") ;
	}

	init(l, r);
}

CCombinedKernel::~CCombinedKernel()
{
	delete[] subkernel_weights_buffer;
	subkernel_weights_buffer=NULL;
	
	cleanup();
	delete kernel_list;

	SG_INFO("Combined kernel deleted (%p).\n", this);
}

bool CCombinedKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l,r);
	ASSERT(l->get_feature_class()==C_COMBINED);
	ASSERT(r->get_feature_class()==C_COMBINED);
	ASSERT(l->get_feature_type()==F_UNKNOWN);
	ASSERT(r->get_feature_type()==F_UNKNOWN);

	CFeatures* lf=NULL;
	CFeatures* rf=NULL;
	CKernel* k=NULL;

	bool result=true;

	CListElement<CFeatures*>*lfc = NULL, *rfc = NULL ;
	lf=((CCombinedFeatures*) l)->get_first_feature_obj(lfc) ;
	rf=((CCombinedFeatures*) r)->get_first_feature_obj(rfc) ;
	CListElement<CKernel*>* current = NULL ;
	k=get_first_kernel(current) ;

	result = 1 ;
	
	if ( lf && rf && k)
	{
		if (l!=r)
		{
			while ( result && lf && rf && k )
			{
				SG_DEBUG( "Initializing 0x%p - \"%s\"\n", this, k->get_name());
				result=k->init(lf,rf);

				lf=((CCombinedFeatures*) l)->get_next_feature_obj(lfc) ;
				rf=((CCombinedFeatures*) r)->get_next_feature_obj(rfc) ;
				k=get_next_kernel(current) ;
			}
		}
		else
		{
			while ( result && lf && k )
			{
				SG_DEBUG( "Initializing 0x%p - \"%s\"\n", this, k->get_name());
				result=k->init(lf,rf);

				lf=((CCombinedFeatures*) l)->get_next_feature_obj(lfc) ;
				k=get_next_kernel(current) ;
				rf=lf ;
			}
		}
	}

	if (!result)
	{
		SG_INFO( "CombinedKernel: Initialising the following kernel failed\n");
		if (k)
			k->list_kernel();
		else
			SG_INFO( "<NULL>\n");
		return false;
	}

	if ((lf!=NULL) || (rf!=NULL) || (k!=NULL))
	{
		SG_INFO( "CombinedKernel: Number of features/kernels does not match - bailing out\n");
		return false;
	}
	
	init_normalizer();
	return true;
}

void CCombinedKernel::remove_lhs()
{
	delete_optimization();

#ifdef SVMLIGHT
	if (lhs)
		cache_reset() ;
#endif
	lhs=NULL ;
	
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		k->remove_lhs();
		k=get_next_kernel(current);
	}
}

void CCombinedKernel::remove_rhs()
{
#ifdef SVMLIGHT
	if (rhs)
		cache_reset() ;
#endif
	rhs=NULL ;

	CListElement<CKernel*> * current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		k->remove_rhs();
		k=get_next_kernel(current);
	}
}

void CCombinedKernel::cleanup()
{
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		k->cleanup();
		k=get_next_kernel(current);
	}

	delete_optimization();

	CKernel::cleanup();
}

void CCombinedKernel::list_kernels()
{
	CKernel* k;

	SG_INFO( "BEGIN COMBINED KERNEL LIST - ");
	this->list_kernel();

	CListElement<CKernel*> * current = NULL ;	
	k=get_first_kernel(current);
	while (k)
	{
		k->list_kernel();
		k=get_next_kernel(current);
	}
	SG_INFO( "END COMBINED KERNEL LIST - ");
}

float64_t CCombinedKernel::compute(int32_t x, int32_t y)
{
	float64_t result=0;
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k=get_first_kernel(current);
	while (k)
	{
		if (k->get_combined_kernel_weight()!=0)
			result += k->get_combined_kernel_weight() * k->kernel(x,y);
		k=get_next_kernel(current);
	}

	return result;
}

bool CCombinedKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t *weights)
{
	SG_DEBUG( "initializing CCombinedKernel optimization\n");

	delete_optimization();

	CListElement<CKernel*> *current=NULL;
	CKernel *k=get_first_kernel(current);
	bool have_non_optimizable=false;

	while(k)
	{
		bool ret=true;

		if (k && k->has_property(KP_LINADD))
			ret=k->init_optimization(count, IDX, weights);
		else
		{
			SG_WARNING("non-optimizable kernel 0x%X in kernel-list\n", k);
			have_non_optimizable=true;
		}
		
		if (!ret)
		{
			have_non_optimizable=true;
			SG_WARNING("init_optimization of kernel 0x%X failed\n", k);
		}
		
		k=get_next_kernel(current);
	}
	
	if (have_non_optimizable)
	{
		SG_WARNING( "some kernels in the kernel-list are not optimized\n");

		sv_idx=new int32_t[count];
		sv_weight=new float64_t[count];
		sv_count=count;
		for (int32_t i=0; i<count; i++)
		{
			sv_idx[i]=IDX[i];
			sv_weight[i]=weights[i];
		}
	}
	set_is_initialized(true);

	return true;
}

bool CCombinedKernel::delete_optimization() 
{ 
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		if (k->has_property(KP_LINADD))
			k->delete_optimization();

		k = get_next_kernel(current);
	}

	delete[] sv_idx;
	sv_idx = NULL;

	delete[] sv_weight;
	sv_weight = NULL;

	sv_count = 0;
	set_is_initialized(false);

	return true;
}

void CCombinedKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* result, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
	ASSERT(rhs);
	ASSERT(num_vec<=rhs->get_num_vectors())
	ASSERT(num_vec>0);
	ASSERT(vec_idx);
	ASSERT(result);

	//we have to do the optimization business ourselves but lets
	//make sure we start cleanly
	delete_optimization();

	CListElement<CKernel*> * current = NULL ;	
	CKernel * k = get_first_kernel(current) ;

	while(k)
	{
		if (k && k->has_property(KP_BATCHEVALUATION))
		{
			if (k->get_combined_kernel_weight()!=0)
				k->compute_batch(num_vec, vec_idx, result, num_suppvec, IDX, weights, k->get_combined_kernel_weight());
		}
		else
			emulate_compute_batch(k, num_vec, vec_idx, result, num_suppvec, IDX, weights);

		k = get_next_kernel(current);
	}

	//clean up
	delete_optimization();
}

void* CCombinedKernel::compute_optimized_kernel_helper(void* p)
{
	S_THREAD_PARAM* params= (S_THREAD_PARAM*) p;
	int32_t* vec_idx=params->vec_idx;
	CKernel* k=params->kernel;
	float64_t* result=params->result;

	for (int32_t i=params->start; i<params->end; i++)
		result[i] += k->get_combined_kernel_weight()*k->compute_optimized(vec_idx[i]);

	return NULL;
}

void* CCombinedKernel::compute_kernel_helper(void* p)
{
	S_THREAD_PARAM* params= (S_THREAD_PARAM*) p;
	int32_t* vec_idx=params->vec_idx;
	CKernel* k=params->kernel;
	float64_t* result=params->result;
	float64_t* weights=params->weights;
	int32_t* IDX=params->IDX;
	int32_t num_suppvec=params->num_suppvec;

	for (int32_t i=params->start; i<params->end; i++)
	{
		float64_t sub_result=0;
		for (int32_t j=0; j<num_suppvec; j++)
			sub_result += weights[j] * k->kernel(IDX[j], vec_idx[i]);

		result[i] += k->get_combined_kernel_weight()*sub_result;
	}

	return NULL;
}

void CCombinedKernel::emulate_compute_batch(
	CKernel* k, int32_t num_vec, int32_t* vec_idx, float64_t* result,
	int32_t num_suppvec, int32_t* IDX, float64_t* weights)
{
	ASSERT(k);
	ASSERT(result);

	if (k->has_property(KP_LINADD))
	{
		if (k->get_combined_kernel_weight()!=0)
		{
			k->init_optimization(num_suppvec, IDX, weights);

			int32_t num_threads=parallel.get_num_threads();
			ASSERT(num_threads>0);

			if (num_threads < 2)
			{
				S_THREAD_PARAM params;
				params.kernel=k;
				params.result=result;
				params.start=0;
				params.end=num_vec;
				params.vec_idx = vec_idx;
				compute_optimized_kernel_helper((void*) &params);
			}
#ifndef WIN32
			else
			{
				pthread_t* threads = new pthread_t[num_threads-1];
				S_THREAD_PARAM* params = new S_THREAD_PARAM[num_threads];
				int32_t step= num_vec/num_threads;

				int32_t t;

				for (t=0; t<num_threads-1; t++)
				{
					params[t].kernel = k;
					params[t].result = result;
					params[t].start = t*step;
					params[t].end = (t+1)*step;
					params[t].vec_idx = vec_idx;
					pthread_create(&threads[t], NULL, CCombinedKernel::compute_optimized_kernel_helper, (void*)&params[t]);
				}

				params[t].kernel = k;
				params[t].result = result;
				params[t].start = t*step;
				params[t].end = num_vec;
				params[t].vec_idx = vec_idx;
				compute_optimized_kernel_helper((void*) &params[t]);

				for (t=0; t<num_threads-1; t++)
					pthread_join(threads[t], NULL);

				delete[] params;
				delete[] threads;
			}
#endif

			k->delete_optimization();
		}
	}
	else
	{
		ASSERT(IDX!=NULL || num_suppvec==0);
		ASSERT(weights!=NULL || num_suppvec==0);

		if (k->get_combined_kernel_weight()!=0)
		{ // compute the usual way for any non-optimized kernel
			int32_t num_threads=parallel.get_num_threads();
			ASSERT(num_threads>0);

			if (num_threads < 2)
			{
				S_THREAD_PARAM params;
				params.kernel=k;
				params.result=result;
				params.start=0;
				params.end=num_vec;
				params.vec_idx = vec_idx;
				params.IDX = IDX;
				params.weights = weights;
				params.num_suppvec = num_suppvec;
				compute_kernel_helper((void*) &params);
			}
#ifndef WIN32
			else
			{
				pthread_t* threads = new pthread_t[num_threads-1];
				S_THREAD_PARAM* params = new S_THREAD_PARAM[num_threads];
				int32_t step= num_vec/num_threads;

				int32_t t;

				for (t=0; t<num_threads-1; t++)
				{
					params[t].kernel = k;
					params[t].result = result;
					params[t].start = t*step;
					params[t].end = (t+1)*step;
					params[t].vec_idx = vec_idx;
					params[t].IDX = IDX;
					params[t].weights = weights;
					params[t].num_suppvec = num_suppvec;
					pthread_create(&threads[t], NULL, CCombinedKernel::compute_kernel_helper, (void*)&params[t]);
				}

				params[t].kernel = k;
				params[t].result = result;
				params[t].start = t*step;
				params[t].end = num_vec;
				params[t].vec_idx = vec_idx;
				params[t].IDX = IDX;
				params[t].weights = weights;
				params[t].num_suppvec = num_suppvec;
				compute_kernel_helper(&params[t]);

				for (t=0; t<num_threads-1; t++)
					pthread_join(threads[t], NULL);

				delete[] params;
				delete[] threads;
			}
#endif
		}
	}
}

float64_t CCombinedKernel::compute_optimized(int32_t idx)
{ 		  
	if (!get_is_initialized())
	{
		SG_ERROR("CCombinedKernel optimization not initialized\n");
		return 0;
	}
	
	float64_t result=0;

	CListElement<CKernel*> *current=NULL;
	CKernel *k=get_first_kernel(current);
	while(k)
	{
		if (k && k->has_property(KP_LINADD) &&
			k->get_is_initialized())
		{
			if (k->get_combined_kernel_weight()!=0)
				result +=
					k->get_combined_kernel_weight()*k->compute_optimized(idx);
		}
		else
		{
			ASSERT(sv_idx!=NULL || sv_count==0);
			ASSERT(sv_weight!=NULL || sv_count==0);

			if (k->get_combined_kernel_weight()!=0)
			{ // compute the usual way for any non-optimized kernel
				float64_t sub_result=0;
				for (int32_t j=0; j<sv_count; j++)
					sub_result += sv_weight[j] * k->kernel(sv_idx[j], idx);

				result += k->get_combined_kernel_weight()*sub_result;
			}
		}

		k=get_next_kernel(current);
	}

	return result;
}

void CCombinedKernel::add_to_normal(int32_t idx, float64_t weight)
{ 
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		k->add_to_normal(idx, weight);
		k = get_next_kernel(current);
	}
	set_is_initialized(true) ;
}

void CCombinedKernel::clear_normal() 
{ 
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		k->clear_normal() ;
		k = get_next_kernel(current);
	}
	set_is_initialized(true) ;
}

void CCombinedKernel::compute_by_subkernel(
	int32_t idx, float64_t * subkernel_contrib)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			int32_t num = -1 ;
			k->get_subkernel_weights(num);
			if (num>1)
				k->compute_by_subkernel(idx, &subkernel_contrib[i]) ;
			else
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;

			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			if (k->get_combined_kernel_weight()!=0)
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;
			k = get_next_kernel(current);
			i++ ;
		}
	}
}

const float64_t* CCombinedKernel::get_subkernel_weights(int32_t& num_weights)
{
	num_weights = get_num_subkernels() ;
	delete[] subkernel_weights_buffer ;
	subkernel_weights_buffer = new float64_t[num_weights] ;

	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			int32_t num = -1 ;
			const float64_t *w = k->get_subkernel_weights(num);
			ASSERT(num==k->get_num_subkernels());
			for (int32_t j=0; j<num; j++)
				subkernel_weights_buffer[i+j]=w[j] ;
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			subkernel_weights_buffer[i] = k->get_combined_kernel_weight();
			k = get_next_kernel(current);
			i++ ;
		}
	}
	
	return subkernel_weights_buffer ;
}

void CCombinedKernel::set_subkernel_weights(
	float64_t* weights, int32_t num_weights)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			int32_t num = k->get_num_subkernels() ;
			k->set_subkernel_weights(&weights[i],num);
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			k->set_combined_kernel_weight(weights[i]);
			k = get_next_kernel(current);
			i++ ;
		}
	}
}

void CCombinedKernel::set_optimization_type(EOptimizationType t)
{ 
	CKernel* k = get_first_kernel();

	while(k)
	{
		k->set_optimization_type(t);
		k = get_next_kernel(k);
	}

	CKernel::set_optimization_type(t);
}

bool CCombinedKernel::precompute_subkernels()
{
	CKernel* k = get_first_kernel();

	if (!k)
		return false;

	CList<CKernel*>* new_kernel_list = new CList<CKernel*>(true);

	while(k)
	{
		new_kernel_list->append_element(new CCustomKernel(k));
		k = get_next_kernel(k);
	}

	delete kernel_list;
	new_kernel_list=kernel_list;

	return true;
}
