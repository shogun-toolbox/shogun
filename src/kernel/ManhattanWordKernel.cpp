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
#include "kernel/ManhattanWordKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CManhattanWordKernel::CManhattanWordKernel(INT size, DREAL w)
	: CStringKernel<WORD>(size), width(w)
{
	SG_DEBUG( "CManhattanWordKernel with cache size: %d width: %f created\n", size, width);
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);
}

CManhattanWordKernel::CManhattanWordKernel(
	CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, DREAL w)
	: CStringKernel<WORD>(10), width(w)
{
	SG_DEBUG( "CManhattanWordKernel with cache size: %d width: %f created\n", 10, width);
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);

	init(l, r);
}

CManhattanWordKernel::~CManhattanWordKernel()
{
	cleanup();

	delete[] dictionary_weights;
}
  
bool CManhattanWordKernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CStringKernel<WORD>::init(l,r);
	return result;
}

void CManhattanWordKernel::cleanup()
{
}

bool CManhattanWordKernel::load_init(FILE* src)
{
	return false;
}

bool CManhattanWordKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CManhattanWordKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
	WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen==blen);

	INT result=0;

	INT left_idx=0;
	INT right_idx=0;

	//hamming of: if words appear in both vectors _the same number_ of times
	while (left_idx < alen && right_idx < blen)
	{
		WORD sym=avec[left_idx];
		if (avec[left_idx]==bvec[right_idx])
		{
			INT old_left_idx=left_idx;
			INT old_right_idx=right_idx;

			while (left_idx< alen && avec[left_idx]==sym)
				left_idx++;

			while (right_idx< blen && bvec[right_idx]==sym)
				right_idx++;

			result += CMath::abs( (left_idx-old_left_idx) - (right_idx-old_right_idx) );
		}
		else if (avec[left_idx]<bvec[right_idx])
		{

			while (left_idx< alen && avec[left_idx]==sym)
			{
				result++;
				left_idx++;
			}
		}
		else
		{
			sym=bvec[right_idx];

			while (right_idx< blen && bvec[right_idx]==sym)
			{
				result++;
				right_idx++;
			}
		}
	}
	
	result+=blen-right_idx + alen-left_idx;

	return exp(-result/width);
}

