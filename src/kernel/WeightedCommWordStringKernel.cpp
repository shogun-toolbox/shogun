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
	ASSERT(sign == false);
}

CWeightedCommWordStringKernel::CWeightedCommWordStringKernel(CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, bool sign, ENormalizationType n, INT size)
  : CCommWordStringKernel(size, sign, n), weights(NULL)
{
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
