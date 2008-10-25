/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(INT size, bool us)
: CCommWordStringKernel(size, us), degree(0), weights(NULL)
{
	init_dictionary(1<<(sizeof(uint16_t)*9));
	ASSERT(us==false);
}

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(
	CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, bool us, INT size)
: CCommWordStringKernel(size, us), degree(0), weights(NULL)
{
	init_dictionary(1<<(sizeof(uint16_t)*9));
	ASSERT(us==false);

	init(l,r);
}

CWeightedCommWordStringKernel::~CWeightedCommWordStringKernel()
{
	delete[] weights;
}

bool CWeightedCommWordStringKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(((CStringFeatures<uint16_t>*) l)->get_order() ==
			((CStringFeatures<uint16_t>*) r)->get_order());
	degree=((CStringFeatures<uint16_t>*) l)->get_order();
	set_wd_weights();

	CCommWordStringKernel::init(l,r);
	return init_normalizer();
}

void CWeightedCommWordStringKernel::cleanup()
{
	delete[] weights;
	weights=NULL;

	CCommWordStringKernel::cleanup();
}
bool CWeightedCommWordStringKernel::set_wd_weights()
{
	delete[] weights;
	weights=new DREAL[degree];

	INT i;
	DREAL sum=0;
	for (i=0; i<degree; i++)
	{
		weights[i]=degree-i;
		sum+=weights[i];
	}
	for (i=0; i<degree; i++)
		weights[i]/=sum;

	return weights!=NULL;
}

bool CWeightedCommWordStringKernel::set_weights(DREAL* w, INT d)
{
	ASSERT(d==degree);

	delete[] weights;
	weights=new DREAL[degree];
	for (INT i=0; i<degree; i++)
		weights[i]=w[i];
	return true;
}
  
DREAL CWeightedCommWordStringKernel::compute_helper(INT idx_a, INT idx_b, bool do_sort)
{
	INT alen, blen;

	CStringFeatures<uint16_t>* l = (CStringFeatures<uint16_t>*) lhs;
	CStringFeatures<uint16_t>* r = (CStringFeatures<uint16_t>*) rhs;

	uint16_t* av=l->get_feature_vector(idx_a, alen);
	uint16_t* bv=r->get_feature_vector(idx_b, blen);

	uint16_t* avec=av;
	uint16_t* bvec=bv;

	if (do_sort)
	{
		if (alen>0)
		{
			avec=new uint16_t[alen];
			memcpy(avec, av, sizeof(uint16_t)*alen);
			CMath::radix_sort(avec, alen);
		}
		else
			avec=NULL;

		if (blen>0)
		{
			bvec=new uint16_t[blen];
			memcpy(bvec, bv, sizeof(uint16_t)*blen);
			CMath::radix_sort(bvec, blen);
		}
		else
			bvec=NULL;
	}
	else
	{
		if ( (l->get_num_preproc() != l->get_num_preprocessed()) ||
				(r->get_num_preproc() != r->get_num_preprocessed()))
		{
			SG_ERROR("not all preprocessors have been applied to training (%d/%d)"
					" or test (%d/%d) data\n", l->get_num_preprocessed(), l->get_num_preproc(),
					r->get_num_preprocessed(), r->get_num_preproc());
		}
	}

	DREAL result=0;
	uint8_t mask=0;

	for (INT d=0; d<degree; d++)
	{
		mask = mask | (1 << (degree-d-1));
		uint16_t masked=((CStringFeatures<uint16_t>*) lhs)->get_masked_symbols(0xffff, mask);

		INT left_idx=0;
		INT right_idx=0;

		while (left_idx < alen && right_idx < blen)
		{
			uint16_t lsym=avec[left_idx] & masked;
			uint16_t rsym=bvec[right_idx] & masked;

			if (lsym == rsym)
			{
				INT old_left_idx=left_idx;
				INT old_right_idx=right_idx;

				while (left_idx<alen && (avec[left_idx] & masked) ==lsym)
					left_idx++;

				while (right_idx<blen && (bvec[right_idx] & masked) ==lsym)
					right_idx++;

				result+=weights[d]*(left_idx-old_left_idx)*(right_idx-old_right_idx);
			}
			else if (lsym<rsym)
				left_idx++;
			else
				right_idx++;
		}
	}

	if (do_sort)
	{
		delete[] avec;
		delete[] bvec;
	}

	return result;
}

void CWeightedCommWordStringKernel::add_to_normal(INT vec_idx, DREAL weight)
{
	INT len=-1;
	CStringFeatures<uint16_t>* s=(CStringFeatures<uint16_t>*) lhs;
	uint16_t* vec=s->get_feature_vector(vec_idx, len);

	if (len>0)
	{
		for (INT j=0; j<len; j++)
		{
			uint8_t mask=0;
			INT offs=0;
			for (INT d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				INT idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				dictionary_weights[offs + idx] += normalizer->normalize_lhs(weight*weights[d], vec_idx);
				offs+=s->shift_offset(1,d+1);
			}
		}

		set_is_initialized(true);
	}
}

void CWeightedCommWordStringKernel::merge_normal()
{
	ASSERT(get_is_initialized());
	ASSERT(use_sign==false);

	CStringFeatures<uint16_t>* s=(CStringFeatures<uint16_t>*) rhs;
	uint32_t num_symbols=(uint32_t) s->get_num_symbols();
	INT dic_size=1<<(sizeof(uint16_t)*8);
	DREAL* dic=new DREAL[dic_size];
	memset(dic, 0, sizeof(DREAL)*dic_size);

	for (uint32_t sym=0; sym<num_symbols; sym++)
	{
		DREAL result=0;
		uint8_t mask=0;
		INT offs=0;
		for (INT d=0; d<degree; d++)
		{
			mask = mask | (1 << (degree-d-1));
			INT idx=s->get_masked_symbols(sym, mask);
			idx=s->shift_symbol(idx, degree-d-1);
			result += dictionary_weights[offs + idx];
			offs+=s->shift_offset(1,d+1);
		}
		dic[sym]=result;
	}

	init_dictionary(1<<(sizeof(uint16_t)*8));
	memcpy(dictionary_weights, dic, sizeof(DREAL)*dic_size);
	delete[] dic;
}

DREAL CWeightedCommWordStringKernel::compute_optimized(INT i) 
{ 
	if (!get_is_initialized())
		SG_ERROR( "CCommWordStringKernel optimization not initialized\n");

	ASSERT(use_sign==false);

	DREAL result=0;
	INT len=-1;
	CStringFeatures<uint16_t>* s=(CStringFeatures<uint16_t>*) rhs;
	uint16_t* vec=s->get_feature_vector(i, len);

	if (vec && len>0)
	{
		for (INT j=0; j<len; j++)
		{
			uint8_t mask=0;
			INT offs=0;
			for (INT d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				INT idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				result += dictionary_weights[offs + idx];
				offs+=s->shift_offset(1,d+1);
			}
		}

		result=normalizer->normalize_rhs(result, i);
	}
	return result;
}

DREAL* CWeightedCommWordStringKernel::compute_scoring(INT max_degree, INT& num_feat,
		INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, bool do_init)
{
	if (do_init)
		CCommWordStringKernel::init_optimization(num_suppvec, IDX, alphas);

	INT dic_size=1<<(sizeof(uint16_t)*9);
	DREAL* dic=new DREAL[dic_size];
	memcpy(dic, dictionary_weights, sizeof(DREAL)*dic_size);

	merge_normal();
	DREAL* result=CCommWordStringKernel::compute_scoring(max_degree, num_feat,
			num_sym, target, num_suppvec, IDX, alphas, false);

	init_dictionary(1<<(sizeof(uint16_t)*9));
	memcpy(dictionary_weights,dic,  sizeof(DREAL)*dic_size);
	delete[] dic;

	return result;
}
