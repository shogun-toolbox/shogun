/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/CommUlongStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CCommUlongStringKernel::CCommUlongStringKernel(LONG size, bool use_sign_, 
											 ENormalizationType normalization_) 
  : CStringKernel<ULONG>(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	use_sign(use_sign_), normalization(normalization_)
{
	properties |= KP_LINADD;
	clear_normal();
}

CCommUlongStringKernel::~CCommUlongStringKernel() 
{
	cleanup();
}
  
void CCommUlongStringKernel::remove_lhs() 
{ 
	delete_optimization();

#ifdef SVMLIGHT
	if (lhs)
		cache_reset();
#endif

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;

	lhs = NULL ; 
	rhs = NULL ; 
	initialized = false;
	sqrtdiag_lhs = NULL;
	sqrtdiag_rhs = NULL;
}

void CCommUlongStringKernel::remove_rhs()
{
#ifdef SVMLIGHT
	if (rhs)
		cache_reset();
#endif

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs;
	rhs = lhs;
}

bool CCommUlongStringKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CStringKernel<ULONG>::init(l,r,do_init);
	initialized = false;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
	  delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	sqrtdiag_lhs= new DREAL[lhs->get_num_vectors()];

	for (i=0; i<lhs->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;

	if (l==r)
		sqrtdiag_rhs=sqrtdiag_lhs;
	else
	{
		sqrtdiag_rhs= new DREAL[rhs->get_num_vectors()];
		for (i=0; i<rhs->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;
	}

	ASSERT(sqrtdiag_lhs);
	ASSERT(sqrtdiag_rhs);

	this->lhs=(CStringFeatures<ULONG>*) l;
	this->rhs=(CStringFeatures<ULONG>*) l;

	//compute normalize to 1 values
	for (i=0; i<lhs->get_num_vectors(); i++)
	{
		sqrtdiag_lhs[i]=sqrt(compute(i,i));

		//trap divide by zero exception
		if (sqrtdiag_lhs[i]==0)
			sqrtdiag_lhs[i]=1e-16;
	}

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs=(CStringFeatures<ULONG>*) r;
		this->rhs=(CStringFeatures<ULONG>*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
			sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=(CStringFeatures<ULONG>*) l;
	this->rhs=(CStringFeatures<ULONG>*) r;

	initialized = true;
	return result;
}

void CCommUlongStringKernel::cleanup()
{
	delete_optimization();
	clear_normal();

	initialized=false;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;
}

bool CCommUlongStringKernel::load_init(FILE* src)
{
	return false;
}

bool CCommUlongStringKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CCommUlongStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	ULONG* avec=((CStringFeatures<ULONG>*) lhs)->get_feature_vector(idx_a, alen);
	ULONG* bvec=((CStringFeatures<ULONG>*) rhs)->get_feature_vector(idx_b, blen);

	DREAL result=0;

	INT left_idx=0;
	INT right_idx=0;

	if (use_sign)
	{
		while (left_idx < alen && right_idx < blen)
		{
			if (avec[left_idx]==bvec[right_idx])
			{
				ULONG sym=avec[left_idx];

				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;

				result++;
			}
			else if (avec[left_idx]<bvec[right_idx])
				left_idx++;
			else
				right_idx++;
		}
	}
	else
	{
		while (left_idx < alen && right_idx < blen)
		{
			if (avec[left_idx]==bvec[right_idx])
			{
				INT old_left_idx=left_idx;
				INT old_right_idx=right_idx;

				ULONG sym=avec[left_idx];

				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;

				result+=(left_idx-old_left_idx)*(right_idx-old_right_idx);
			}
			else if (avec[left_idx]<bvec[right_idx])
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
				CIO::message(M_ERROR, "Unknown Normalization in use!\n");
				return -CMath::INFTY;
		}
	}
	else
		return result;
}

void CCommUlongStringKernel::add_to_normal(INT vec_idx, DREAL weight)
{
	INT t=0;
	INT j=0;
	INT k=0;
	INT last_j=0;
	INT len=-1;
	ULONG* vec=((CStringFeatures<ULONG>*) lhs)->get_feature_vector(vec_idx, len);

	if (vec && len>0)
	{
		//use malloc not new [] as DynamicArray uses it
		ULONG* dic= (ULONG*) malloc((len+dictionary.get_num_elements())*sizeof(ULONG));
		DREAL* dic_weights= (DREAL*) malloc((len+dictionary.get_num_elements())*sizeof(DREAL));

		if (use_sign)
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				merge_dictionaries(t, j, k, vec, dic, dic_weights, weight, vec_idx, len, normalization);
			}

			merge_dictionaries(t, j, k, vec, dic, dic_weights, weight, vec_idx, len, normalization);

			while (k<dictionary.get_num_elements())
			{
				dic[t]=dictionary[k];
				dic_weights[t]=dictionary_weights[k];
				t++;
				k++;
			}
		}
		else
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				merge_dictionaries(t, j, k, vec, dic, dic_weights, weight*(j-last_j), vec_idx, len, normalization);
				last_j = j;
			}

			merge_dictionaries(t, j, k, vec, dic, dic_weights, weight*(j-last_j), vec_idx, len, normalization);

			while (k<dictionary.get_num_elements())
			{
				dic[t]=dictionary[k];
				dic_weights[t]=dictionary_weights[k];
				t++;
				k++;
			}
		}

		dictionary.set_array(dic, t, len+dictionary.get_num_elements());
		dictionary_weights.set_array(dic_weights, t, len+dictionary.get_num_elements());
	}

	set_is_initialized(true);
}

void CCommUlongStringKernel::clear_normal()
{
	dictionary.resize_array(0);
	dictionary_weights.resize_array(0);
	set_is_initialized(false);
}

bool CCommUlongStringKernel::init_optimization(INT count, INT *IDX, DREAL * weights) 
{
	clear_normal();

	if (count<=0)
	{
		set_is_initialized(true);
		CIO::message(M_DEBUG, "empty set of SVs\n");
		return true;
	}

	CIO::message(M_DEBUG, "initializing CCommUlongStringKernel optimization\n");

	for (int i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::progress(i, 0, count);

		add_to_normal(IDX[i], weights[i]);
	}

	CIO::message(M_MESSAGEONLY, "Done.         \n");
	
	set_is_initialized(true);
	return true;
}

bool CCommUlongStringKernel::delete_optimization() 
{
	CIO::message(M_DEBUG, "deleting CCommUlongStringKernel optimization\n");
	clear_normal();
	return true;
}

// binary search for each feature. trick: as features are sorted save last found idx in old_idx and
// only search in the remainder of the dictionary
DREAL CCommUlongStringKernel::compute_optimized(INT i) 
{ 
	DREAL result = 0;
	INT j, last_j=0;
	INT old_idx = 0;

	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCommUlongStringKernel optimization not initialized\n");
		return 0 ; 
	}



	INT alen = -1;
	ULONG* avec=((CStringFeatures<ULONG>*) rhs)->get_feature_vector(i, alen);

	if (avec && alen>0)
	{
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue;

				INT idx = CMath::binary_search_max_lower_equal(&(dictionary.get_array()[old_idx]), dictionary.get_num_elements()-old_idx, avec[j-1]);

				if (idx!=-1)
				{
					if (dictionary[idx+old_idx] == avec[j-1])
						result += dictionary_weights[idx+old_idx];

					old_idx+=idx;
				}
			}

			INT idx = CMath::binary_search(&(dictionary.get_array()[old_idx]), dictionary.get_num_elements()-old_idx, avec[alen-1]);
			if (idx!=-1)
				result += dictionary_weights[idx+old_idx];
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue;

				INT idx = CMath::binary_search_max_lower_equal(&(dictionary.get_array()[old_idx]), dictionary.get_num_elements()-old_idx, avec[j-1]);

				if (idx!=-1)
				{
					if (dictionary[idx+old_idx] == avec[j-1])
						result += dictionary_weights[idx+old_idx]*(j-last_j);

					old_idx+=idx;
				}

				last_j = j;
			}

			INT idx = CMath::binary_search(&(dictionary.get_array()[old_idx]), dictionary.get_num_elements()-old_idx, avec[alen-1]);
			if (idx!=-1)
				result += dictionary_weights[idx+old_idx]*(alen-last_j);
		}

		switch (normalization)
		{
			case NO_NORMALIZATION:
				return result;
			case SQRT_NORMALIZATION:
				return result/sqrt(sqrtdiag_rhs[i]);
			case FULL_NORMALIZATION:
				return result/sqrtdiag_rhs[i];
			case SQRTLEN_NORMALIZATION:
				return result/sqrt(sqrt(alen));
			case LEN_NORMALIZATION:
				return result/sqrt(alen);
			case SQLEN_NORMALIZATION:
				return result/alen;
			default:
				CIO::message(M_ERROR, "Unknown Normalization in use!\n");
				return -CMath::INFTY;
		}
	}
	return result;
}
