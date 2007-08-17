/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(LONG size, bool sign, ENormalizationType n)
  : CCommWordStringKernel(size, sign, n), weights(NULL)
{
	init_dictionary(1<<(sizeof(WORD)*9));
	ASSERT(sign == false);
}

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, bool sign, ENormalizationType n, INT size)
  : CCommWordStringKernel(size, sign, n), weights(NULL)
{
	init_dictionary(1<<(sizeof(WORD)*9));
	ASSERT(sign == false);
	init(l,r);
}

CWeightedCommWordStringKernel::~CWeightedCommWordStringKernel() 
{
}

bool CWeightedCommWordStringKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(((CStringFeatures<WORD>*) l)->get_order() ==
			((CStringFeatures<WORD>*) r)->get_order());
	degree=((CStringFeatures<WORD>*) l)->get_order();
	set_wd_weights();

	return CCommWordStringKernel::init(l,r);
}

void CWeightedCommWordStringKernel::cleanup()
{
	delete[] weights;
	weights=NULL;

	CCommWordStringKernel::cleanup();
}
bool CWeightedCommWordStringKernel::set_wd_weights()
{
	SG_DEBUG("WSPEC degree: %d\n", degree);
	delete[] weights;
	weights=new DREAL[degree];
	ASSERT(weights);

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
  
DREAL CWeightedCommWordStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
	WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);

	DREAL result=0;

	BYTE mask=0;

	for (INT d=0; d<degree; d++)
	{
		mask = mask | (1 << (degree-d-1));
		WORD masked=((CStringFeatures<WORD>*) lhs)->get_masked_symbols(0xffff, mask);

		INT left_idx=0;
		INT right_idx=0;

		while (left_idx < alen && right_idx < blen)
		{
			WORD lsym=avec[left_idx] & masked;
			WORD rsym=bvec[right_idx] & masked;

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

	if (initialized)
	{
		switch (normalization)
		{
			case NO_NORMALIZATION:
				return result;
			case SQRT_NORMALIZATION:
				return result/sqrt(sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]);
			case FULL_NORMALIZATION:
				return result/(sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]);
			case SQRTLEN_NORMALIZATION:
				return result/sqrt(sqrt(alen*blen));
			case LEN_NORMALIZATION:
				return result/sqrt(alen*blen);
			case SQLEN_NORMALIZATION:
				return result/(alen*blen);
			default:
				SG_ERROR( "Unknown Normalization in use!\n");
				return -CMath::INFTY;
		}
	}
	else
		return result;
}

void CWeightedCommWordStringKernel::add_to_normal(INT vec_idx, DREAL weight)
{
	INT len=-1;
	CStringFeatures<WORD>* s=(CStringFeatures<WORD>*) lhs;
	WORD* vec=s->get_feature_vector(vec_idx, len);

	if (len>0)
	{
		for (INT j=0; j<len; j++)
		{
			BYTE mask=0;
			INT offs=0;
			for (INT d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				INT idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				dictionary_weights[offs + idx] += normalize_weight(sqrtdiag_lhs, weight*weights[d], vec_idx, len, normalization);
				offs+=s->shift_offset(1,d+1);
			}
		}

		set_is_initialized(true);
	}
}

void CWeightedCommWordStringKernel::merge_normal()
{
	ASSERT(get_is_initialized());
	ASSERT(use_sign == false);
	CStringFeatures<WORD>* s=(CStringFeatures<WORD>*) rhs;
	UINT num_symbols=(UINT) s->get_num_symbols();
	INT dic_size=1<<(sizeof(WORD)*8);
	DREAL* dic= new DREAL[dic_size];
	ASSERT(dic);
	memset(dic, 0, sizeof(DREAL)*dic_size);

	for (UINT sym=0; sym<num_symbols; sym++)
	{
		DREAL result=0;
		BYTE mask=0;
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

	init_dictionary(1<<(sizeof(WORD)*8));
	memcpy(dictionary_weights, dic, sizeof(DREAL)*dic_size);
	delete[] dic;
}

DREAL CWeightedCommWordStringKernel::compute_optimized(INT i) 
{ 
	if (!get_is_initialized())
	{
		SG_ERROR( "CCommWordStringKernel optimization not initialized\n");
		return 0 ; 
	}

	ASSERT(use_sign == false);

	DREAL result = 0;
	INT len = -1;

	CStringFeatures<WORD>* s=(CStringFeatures<WORD>*) rhs;
	WORD* vec=s->get_feature_vector(i, len);

	if (vec && len>0)
	{
		for (INT j=0; j<len; j++)
		{
			BYTE mask=0;
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

		result=normalize_weight(sqrtdiag_rhs, result, i, len, normalization);
	}
	return result;
}

DREAL* CWeightedCommWordStringKernel::compute_scoring(INT max_degree, INT& num_feat,
		INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, bool do_init)
{
	if (do_init)
		CCommWordStringKernel::init_optimization(num_suppvec, IDX, alphas);

	INT dic_size=1<<(sizeof(WORD)*9);
	DREAL* dic= new DREAL[dic_size];
	ASSERT(dic);
	memcpy(dic, dictionary_weights, sizeof(DREAL)*dic_size);

	merge_normal();
	DREAL* result=CCommWordStringKernel::compute_scoring(max_degree, num_feat,
			num_sym, target, num_suppvec, IDX, alphas, false);

	init_dictionary(1<<(sizeof(WORD)*9));
	memcpy(dictionary_weights,dic,  sizeof(DREAL)*dic_size);
	delete[] dic;

	return result;
}
