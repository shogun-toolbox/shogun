/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Christian Gehl
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "distance/HammingWordDistance.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CHammingWordDistance::CHammingWordDistance(bool sign)
: CStringDistance<uint16_t>(), use_sign(sign)
{
	SG_DEBUG( "CHammingWordDistance with sign: %d created\n", (sign) ? 1 : 0);
	dictionary_size= 1<<(sizeof(uint16_t)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);
}

CHammingWordDistance::CHammingWordDistance(
	CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, bool sign)
: CStringDistance<uint16_t>(), use_sign(sign)
{
	SG_DEBUG( "CHammingWordDistance with sign: %d created\n", (sign) ? 1 : 0);
	dictionary_size= 1<<(sizeof(uint16_t)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);

	init(l, r);
}

CHammingWordDistance::~CHammingWordDistance()
{
	cleanup();

	delete[] dictionary_weights;
}
  
bool CHammingWordDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CStringDistance<uint16_t>::init(l,r);
	return result;
}

void CHammingWordDistance::cleanup()
{
}

bool CHammingWordDistance::load_init(FILE* src)
{
	return false;
}

bool CHammingWordDistance::save_init(FILE* dest)
{
	return false;
}
  
DREAL CHammingWordDistance::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	uint16_t* avec=((CStringFeatures<uint16_t>*) lhs)->get_feature_vector(idx_a, alen);
	uint16_t* bvec=((CStringFeatures<uint16_t>*) rhs)->get_feature_vector(idx_b, blen);

	INT result=0;

	INT left_idx=0;
	INT right_idx=0;

	if (use_sign)
	{
		// hamming of: if words appear in both vectors 
		while (left_idx < alen && right_idx < blen)
		{
			uint16_t sym=avec[left_idx];
			if (avec[left_idx]==bvec[right_idx])
			{
				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;
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
	}
	else
	{
		//hamming of: if words appear in both vectors _the same number_ of times
		while (left_idx < alen && right_idx < blen)
		{
			uint16_t sym=avec[left_idx];
			if (avec[left_idx]==bvec[right_idx])
			{
				INT old_left_idx=left_idx;
				INT old_right_idx=right_idx;

				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;

				if ((left_idx-old_left_idx)!=(right_idx-old_right_idx))
					result++;
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
	}

	while (left_idx < alen)
	{
		uint16_t sym=avec[left_idx];
		result++;

		while (left_idx< alen && avec[left_idx]==sym)
			left_idx++;
	}

	while (right_idx < blen)
	{
		uint16_t sym=bvec[right_idx];
		result++;

		while (right_idx< blen && bvec[right_idx]==sym)
			right_idx++;
	}

	return result;
}
