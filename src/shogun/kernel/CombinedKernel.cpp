/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/CombinedFeatures.h>

#include <string.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif // DOXYGEN_SHOULD_SKIP_THIS

CCombinedKernel::CCombinedKernel(int32_t size, bool asw)
: CKernel(size), append_subkernel_weights(asw)
{
	init();

	if (append_subkernel_weights)
		SG_INFO( "(subkernel weights are appended)\n") ;

	SG_INFO("Combined kernel created (%p)\n", this) ;
}

CCombinedKernel::~CCombinedKernel()
{
	SG_FREE(subkernel_weights_buffer);
	subkernel_weights_buffer=NULL;
	
	cleanup();
	SG_UNREF(kernel_list);

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

	CListElement* lfc = NULL;
	CListElement* rfc = NULL;
	lf=((CCombinedFeatures*) l)->get_first_feature_obj(lfc);
	rf=((CCombinedFeatures*) r)->get_first_feature_obj(rfc);
	CListElement* current = NULL;
	k=get_first_kernel(current);

	while ( result && k )
	{
		// skip over features - the custom kernel does not need any
		if (k->get_kernel_type() != K_CUSTOM)
		{
			if (!lf || !rf)
			{
				SG_UNREF(lf);
				SG_UNREF(rf);
				SG_UNREF(k);
				SG_ERROR( "CombinedKernel: Number of features/kernels does not match - bailing out\n");
			}

			SG_DEBUG( "Initializing 0x%p - \"%s\"\n", this, k->get_name());
			result=k->init(lf,rf);
			SG_UNREF(lf);
			SG_UNREF(rf);

			lf=((CCombinedFeatures*) l)->get_next_feature_obj(lfc) ;
			rf=((CCombinedFeatures*) r)->get_next_feature_obj(rfc) ;
		}
		else
		{
			SG_DEBUG( "Initializing 0x%p - \"%s\" (skipping init, this is a CUSTOM kernel)\n", this, k->get_name());
			if (!k->has_features())
				SG_ERROR("No kernel matrix was assigned to this Custom kernel\n");
			if (k->get_num_vec_lhs() != num_lhs)
				SG_ERROR("Number of lhs-feature vectors (%d) not match with number of rows (%d) of custom kernel\n", num_lhs, k->get_num_vec_lhs());
			if (k->get_num_vec_rhs() != num_rhs)
				SG_ERROR("Number of rhs-feature vectors (%d) not match with number of cols (%d) of custom kernel\n", num_rhs, k->get_num_vec_rhs());
		}

		SG_UNREF(k);
		k=get_next_kernel(current) ;
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
		SG_UNREF(lf);
		SG_UNREF(rf);
		SG_UNREF(k);
		SG_ERROR( "CombinedKernel: Number of features/kernels does not match - bailing out\n");
	}
	
	init_normalizer();
	initialized=true;
	return true;
}

void CCombinedKernel::remove_lhs()
{
	delete_optimization();

	CListElement* current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs();

		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	CKernel::remove_lhs();

	num_lhs=0;
}

void CCombinedKernel::remove_rhs()
{
	CListElement* current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_rhs();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	CKernel::remove_rhs();

	num_rhs=0;
}

void CCombinedKernel::remove_lhs_and_rhs()
{
	delete_optimization();

	CListElement* current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs_and_rhs();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	CKernel::remove_lhs_and_rhs();

	num_lhs=0;
	num_rhs=0;
}

void CCombinedKernel::cleanup()
{
	CListElement* current = NULL ;	
	CKernel* k=get_first_kernel(current);

	while (k)
	{	
		k->cleanup();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	delete_optimization();

	CKernel::cleanup();

	num_lhs=0;
	num_rhs=0;
}

void CCombinedKernel::list_kernels()
{
	CKernel* k;

	SG_INFO( "BEGIN COMBINED KERNEL LIST - ");
	this->list_kernel();

	CListElement* current = NULL ;	
	k=get_first_kernel(current);
	while (k)
	{
		k->list_kernel();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	SG_INFO( "END COMBINED KERNEL LIST - ");
}

float64_t CCombinedKernel::compute(int32_t x, int32_t y)
{
	float64_t result=0;
	CListElement* current = NULL ;	
	CKernel* k=get_first_kernel(current);
	while (k)
	{
		if (k->get_combined_kernel_weight()!=0)
			result += k->get_combined_kernel_weight() * k->kernel(x,y);
		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	return result;
}

bool CCombinedKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t *weights)
{
	SG_DEBUG( "initializing CCombinedKernel optimization\n");

	delete_optimization();

	CListElement* current=NULL;
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
		
		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	
	if (have_non_optimizable)
	{
		SG_WARNING( "some kernels in the kernel-list are not optimized\n");

		sv_idx=SG_MALLOC(int32_t, count);
		sv_weight=SG_MALLOC(float64_t, count);
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
	CListElement* current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		if (k->has_property(KP_LINADD))
			k->delete_optimization();

		SG_UNREF(k);
		k = get_next_kernel(current);
	}

	SG_FREE(sv_idx);
	sv_idx = NULL;

	SG_FREE(sv_weight);
	sv_weight = NULL;

	sv_count = 0;
	set_is_initialized(false);

	return true;
}

void CCombinedKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* result, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
	ASSERT(num_vec<=get_num_vec_rhs())
	ASSERT(num_vec>0);
	ASSERT(vec_idx);
	ASSERT(result);

	//we have to do the optimization business ourselves but lets
	//make sure we start cleanly
	delete_optimization();

	CListElement* current = NULL ;	
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

		SG_UNREF(k);
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

			int32_t num_threads=parallel->get_num_threads();
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
				pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
				S_THREAD_PARAM* params = SG_MALLOC(S_THREAD_PARAM, num_threads);
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

				SG_FREE(params);
				SG_FREE(threads);
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
			int32_t num_threads=parallel->get_num_threads();
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
				pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
				S_THREAD_PARAM* params = SG_MALLOC(S_THREAD_PARAM, num_threads);
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

				SG_FREE(params);
				SG_FREE(threads);
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

	CListElement* current=NULL;
	CKernel *k=get_first_kernel(current);
	while (k)
	{
		if (k->has_property(KP_LINADD) &&
			k->get_is_initialized())
		{
			if (k->get_combined_kernel_weight()!=0)
			{
				result +=
					k->get_combined_kernel_weight()*k->compute_optimized(idx);
			}
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

		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	return result;
}

void CCombinedKernel::add_to_normal(int32_t idx, float64_t weight)
{ 
	CListElement* current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		k->add_to_normal(idx, weight);
		SG_UNREF(k);
		k = get_next_kernel(current);
	}
	set_is_initialized(true) ;
}

void CCombinedKernel::clear_normal() 
{ 
	CListElement* current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		k->clear_normal() ;
		SG_UNREF(k);
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
		CListElement* current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			int32_t num = -1 ;
			k->get_subkernel_weights(num);
			if (num>1)
				k->compute_by_subkernel(idx, &subkernel_contrib[i]) ;
			else
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;

			SG_UNREF(k);
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		CListElement* current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			if (k->get_combined_kernel_weight()!=0)
				subkernel_contrib[i] += k->get_combined_kernel_weight() * k->compute_optimized(idx) ;

			SG_UNREF(k);
			k = get_next_kernel(current);
			i++ ;
		}
	}
}

const float64_t* CCombinedKernel::get_subkernel_weights(int32_t& num_weights)
{
	num_weights = get_num_subkernels() ;
	SG_FREE(subkernel_weights_buffer);
	subkernel_weights_buffer = SG_MALLOC(float64_t, num_weights);

	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		CListElement* current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			int32_t num = -1 ;
			const float64_t *w = k->get_subkernel_weights(num);
			ASSERT(num==k->get_num_subkernels());
			for (int32_t j=0; j<num; j++)
				subkernel_weights_buffer[i+j]=w[j] ;

			SG_UNREF(k);
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		CListElement* current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			subkernel_weights_buffer[i] = k->get_combined_kernel_weight();

			SG_UNREF(k);
			k = get_next_kernel(current);
			i++ ;
		}
	}
	
	return subkernel_weights_buffer ;
}

SGVector<float64_t> CCombinedKernel::get_subkernel_weights()
{
	int32_t num=0;
	const float64_t* w=get_subkernel_weights(num);

	return SGVector<float64_t>((float64_t*) w, num);
}

void CCombinedKernel::set_subkernel_weights(SGVector<float64_t> weights)
{
	if (append_subkernel_weights)
	{
		int32_t i=0 ;
		CListElement* current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			int32_t num = k->get_num_subkernels() ;
			ASSERT(i<weights.vlen);
			k->set_subkernel_weights(SGVector<float64_t>(&weights.vector[i],num));

			SG_UNREF(k);
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		int32_t i=0 ;
		CListElement* current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			ASSERT(i<weights.vlen);
			k->set_combined_kernel_weight(weights.vector[i]);

			SG_UNREF(k);
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

		SG_UNREF(k);
		k = get_next_kernel();
	}

	CKernel::set_optimization_type(t);
}

bool CCombinedKernel::precompute_subkernels()
{
	CKernel* k = get_first_kernel();

	if (!k)
		return false;

	CList* new_kernel_list = new CList(true);

	while(k)
	{
		new_kernel_list->append_element(new CCustomKernel(k));

		SG_UNREF(k);
		k = get_next_kernel();
	}

	SG_UNREF(kernel_list);
	kernel_list=new_kernel_list;
	SG_REF(kernel_list);

	return true;
}

void CCombinedKernel::init()
{
	sv_count=0;
	sv_idx=NULL;
	sv_weight=NULL;
	subkernel_weights_buffer=NULL;
	initialized=false;

	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	kernel_list=new CList(true);
	SG_REF(kernel_list);


	m_parameters->add((CSGObject**) &kernel_list, "kernel_list",
					  "List of kernels.");
	m_parameters->add_vector(&sv_idx, &sv_count, "sv_idx",
							 "Support vector index.");
	m_parameters->add_vector(&sv_weight, &sv_count, "sv_weight",
							 "Support vector weights.");
	m_parameters->add(&append_subkernel_weights,
					  "append_subkernel_weights",
					  "If subkernel weights are appended.");
	m_parameters->add(&initialized, "initialized",
					  "Whether kernel is ready to be used.");
}

