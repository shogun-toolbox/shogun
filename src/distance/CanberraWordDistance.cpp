/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) Christian Gehl
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "distance/CanberraWordDistance.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CCanberraWordDistance::CCanberraWordDistance()
	: CStringDistance<WORD>()
{
	SG_DEBUG("CCanberraWordDistance created");
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);
}

CCanberraWordDistance::~CCanberraWordDistance() 
{
	cleanup();

	delete[] dictionary_weights;
}
  
bool CCanberraWordDistance::init(CFeatures* l, CFeatures* r)
{
	return CStringDistance<WORD>::init(l,r);
}

void CCanberraWordDistance::cleanup()
{
}

bool CCanberraWordDistance::load_init(FILE* src)
{
	return false;
}

bool CCanberraWordDistance::save_init(FILE* dest)
{
	return false;
}
  
DREAL CCanberraWordDistance::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
	WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length -> not as WordString
	//ASSERT(alen==blen);

	DREAL result=0;

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

			result += CMath::abs( (DREAL) ((left_idx-old_left_idx) - (right_idx-old_right_idx)) )/
						( (DREAL) ((left_idx-old_left_idx) + (right_idx-old_right_idx)) );
		}
		else if (avec[left_idx]<bvec[right_idx])
		{
			result++;

			while (left_idx< alen && avec[left_idx]==sym)
				left_idx++;
		}
		else
		{
			sym=bvec[right_idx];
			result++;

			while (right_idx< blen && bvec[right_idx]==sym)
				right_idx++;
		}
	}
	
	while (left_idx < alen)
	{
		WORD sym=avec[left_idx];
		result++;

		while (left_idx< alen && avec[left_idx]==sym)
			left_idx++;
	}

	while (right_idx < blen)
	{
		WORD sym=bvec[right_idx];
		result++;

		while (right_idx< blen && bvec[right_idx]==sym)
			right_idx++;
	}

	return result;
}
