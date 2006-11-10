/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "features/CombinedFeatures.h"
#include "lib/io.h"

#include <string.h>

CCombinedKernel::CCombinedKernel(LONG size, bool append_subkernel_weights_)
	: CKernel(size), sv_count(0), sv_idx(NULL), sv_weight(NULL), 
	  subkernel_weights_buffer(NULL), append_subkernel_weights(append_subkernel_weights_)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	kernel_list=new CList<CKernel*>(true);
	CIO::message(M_INFO, "combined kernel created\n") ;
	if (append_subkernel_weights)
		CIO::message(M_INFO, "(subkernel weights are appended)\n") ;
}

CCombinedKernel::~CCombinedKernel() 
{
	delete[] subkernel_weights_buffer ;
	subkernel_weights_buffer=NULL ;
	
	CIO::message(M_INFO, "combined kernel deleted\n");
	cleanup();
	delete kernel_list;
}

bool CCombinedKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l,r, do_init);
	ASSERT(l->get_feature_class() == C_COMBINED);
	ASSERT(r->get_feature_class() == C_COMBINED);
	ASSERT(l->get_feature_type() == F_UNKNOWN);
	ASSERT(r->get_feature_type() == F_UNKNOWN);

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
				//fprintf(stderr,"init kernel 0x%X (0x%X, 0x%X)\n", k, lf, rf) ;
				result=k->init(lf,rf, do_init);

				lf=((CCombinedFeatures*) l)->get_next_feature_obj(lfc) ;
				rf=((CCombinedFeatures*) r)->get_next_feature_obj(rfc) ;
				k=get_next_kernel(current) ;
			} ;
		}
		else
		{
			while ( result && lf && k )
			{
				result=k->init(lf,rf, do_init);

				//fprintf(stderr,"init kernel 0x%X (0x%X, 0x%X): %i\n", k, lf, rf, result) ;

				lf=((CCombinedFeatures*) l)->get_next_feature_obj(lfc) ;
				k=get_next_kernel(current) ;
				rf=lf ;
			} ;
		}
	}
	//fprintf(stderr,"k=0x%X  lf=0x%X  rf=0x%X  result=%i\n", k, lf, rf, result) ;

	if (!result)
	{
		CIO::message(M_INFO, "CombinedKernel: Initialising the following kernel failed\n");
		if (k)
			k->list_kernel();
		else
			CIO::message(M_INFO, "<NULL>\n");
		return false;
	}

	if ((lf!=NULL) || (rf!=NULL) || (k!=NULL))
	{
		CIO::message(M_INFO, "CombinedKernel: Number of features/kernels does not match - bailing out\n");
		return false;
	}
	
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
}

void CCombinedKernel::list_kernels()
{
	CKernel* k;

	CIO::message(M_INFO, "BEGIN COMBINED KERNEL LIST - ");
	this->list_kernel();

	CListElement<CKernel*> * current = NULL ;	
	k=get_first_kernel(current);
	while (k)
	{
		k->list_kernel();
		k=get_next_kernel(current);
	}
	CIO::message(M_INFO, "END COMBINED KERNEL LIST - ");
}

DREAL CCombinedKernel::compute(INT x, INT y)
{
	DREAL result=0;
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

bool CCombinedKernel::init_optimization(INT count, INT *IDX, DREAL * weights) 
{
	CIO::message(M_DEBUG, "initializing CCombinedKernel optimization\n") ;

	delete_optimization();
	
	CListElement<CKernel*> * current = NULL ;	
	CKernel * k = get_first_kernel(current) ;
	bool have_non_optimizable = false ;
	
	while(k)
	{
		bool ret = true ;
		
		if (k && k->has_property(KP_LINADD))
			ret = k->init_optimization(count, IDX, weights) ;
		else
		{
			CIO::message(M_WARN, "non-optimizable kernel 0x%X in kernel-list\n",k) ;
			have_non_optimizable = true ;
		}
		
		if (!ret)
		{
			have_non_optimizable = true ;
			CIO::message(M_WARN, "init_optimization of kernel 0x%X failed\n",k) ;
		} ;
		
		k = get_next_kernel(current) ;
	}
	
	if (have_non_optimizable)
	{
		CIO::message(M_WARN, "some kernels in the kernel-list are not optimized\n") ;

		sv_idx = new INT[count] ;
		sv_weight = new DREAL[count] ;
		sv_count = count ;
		int i ;
		for (i=0; i<count; i++)
		{
			sv_idx[i] = IDX[i] ;
			sv_weight[i] = weights[i] ;
		}
	}
	set_is_initialized(true) ;
	
	return true ;
} ;

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

void CCombinedKernel::compute_batch(INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* weights, DREAL factor)
{
	ASSERT(get_rhs());
	ASSERT(num_vec<=get_rhs()->get_num_vectors())
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

void CCombinedKernel::emulate_compute_batch(CKernel* k, INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* weights)
{
	ASSERT(k);
	ASSERT(result);

	if (k->has_property(KP_LINADD))
	{
		if (k->get_combined_kernel_weight()!=0)
		{
			k->init_optimization(num_suppvec, IDX, weights);

			for (INT i=0; i<num_vec; i++)
				result[i] += k->get_combined_kernel_weight()*k->compute_optimized(vec_idx[i]);

			k->delete_optimization();
		}
	}
	else
	{
		ASSERT(IDX!=NULL || num_suppvec==0) ;
		ASSERT(weights!=NULL || num_suppvec==0) ;

		if (k->get_combined_kernel_weight()!=0)
		{ // compute the usual way for any non-optimized kernel
			for (INT i=0; i<num_vec; i++)
			{
				int j=0;
				DREAL sub_result=0 ;
				for (j=0; j<num_suppvec; j++)
					sub_result += weights[j] * k->kernel(IDX[j], vec_idx[i]) ;

				result[i] += k->get_combined_kernel_weight()*sub_result ;
			}
		}
	}
}

DREAL CCombinedKernel::compute_optimized(INT idx) 
{ 		  
	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCombinedKernel optimization not initialized\n") ;
		return 0 ;
	}
	
	DREAL result = 0 ;
	
	CListElement<CKernel*> * current = NULL ;	
	CKernel * k = get_first_kernel(current) ;
	while(k)
	{
		if (k && k->has_property(KP_LINADD) && 
			k->get_is_initialized())
		{
			if (k->get_combined_kernel_weight()!=0)
				result += k->get_combined_kernel_weight()*
					k->compute_optimized(idx) ;
		}
		else
		{
			ASSERT(sv_idx!=NULL || sv_count==0) ;
			ASSERT(sv_weight!=NULL || sv_count==0) ;

			if (k->get_combined_kernel_weight()!=0)
			{ // compute the usual way for any non-optimized kernel
				int j=0;
				DREAL sub_result=0 ;
				for (j=0; j<sv_count; j++)
					sub_result += sv_weight[j] * k->kernel(sv_idx[j], idx) ;
				
				result += k->get_combined_kernel_weight()*sub_result ;
			}
		} ;
		
		k = get_next_kernel(current) ;
	}

	return result;
}

void CCombinedKernel::add_to_normal(INT idx, DREAL weight) 
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

void CCombinedKernel::compute_by_subkernel(INT idx, DREAL * subkernel_contrib)
{
	if (append_subkernel_weights)
	{
		INT i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			INT num = -1 ;
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
		INT i=0 ;
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

const DREAL* CCombinedKernel::get_subkernel_weights(INT& num_weights)
{
	num_weights = get_num_subkernels() ;
	delete[] subkernel_weights_buffer ;
	subkernel_weights_buffer = new DREAL[num_weights] ;

	if (append_subkernel_weights)
	{
		INT i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			INT num = -1 ;
			const DREAL *w = k->get_subkernel_weights(num);
			ASSERT(num==k->get_num_subkernels()) ;
			for (INT j=0; j<num; j++)
				subkernel_weights_buffer[i+j]=w[j] ;
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		INT i=0 ;
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

void CCombinedKernel::set_subkernel_weights(DREAL* weights, INT num_weights)
{
	if (append_subkernel_weights)
	{
		INT i=0 ;
		CListElement<CKernel*> * current = NULL ;	
		CKernel* k = get_first_kernel(current);
		while(k)
		{
			INT num = k->get_num_subkernels() ;
			k->set_subkernel_weights(&weights[i],num);
			/*DREAL w = 0 ;
			  for (INT j=0; j<num; j++)
			  if (weights[i+j]!=0)
			  w=1 ;
			  k->set_combined_kernel_weight(w);*/
			k = get_next_kernel(current);
			i += num ;
		}
	}
	else
	{
		INT i=0 ;
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
	CListElement<CKernel*> * current = NULL ;	
	CKernel* k = get_first_kernel(current);

	while(k)
	{
		k->set_optimization_type(t);
		k = get_next_kernel(current);
	}

	CKernel::set_optimization_type(t);
}
