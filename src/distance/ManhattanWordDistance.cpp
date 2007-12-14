/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * Written (W) 2007 Christian Gehl
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "distance/ManhattanWordDistance.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CManhattanWordDistance::CManhattanWordDistance()
: CStringDistance<WORD>()
{
	SG_DEBUG("CManhattanWordDistance created");
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);
}

CManhattanWordDistance::CManhattanWordDistance(
	CStringFeatures<WORD>* l, CStringFeatures<WORD>* r)
: CStringDistance<WORD>()
{
	SG_DEBUG("CManhattanWordDistance created");
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new DREAL[dictionary_size];
	SG_DEBUG( "using dictionary of %d bytes\n", dictionary_size);

	init(l, r);
}

CManhattanWordDistance::~CManhattanWordDistance() 
{
	cleanup();

	delete[] dictionary_weights;
}

bool CManhattanWordDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CStringDistance<WORD>::init(l,r);
	return result;
}

void CManhattanWordDistance::cleanup()
{
}

bool CManhattanWordDistance::load_init(FILE* src)
{
	return false;
}

bool CManhattanWordDistance::save_init(FILE* dest)
{
	return false;
}

DREAL CManhattanWordDistance::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
	WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);

	INT result=0;

	INT left_idx=0;
	INT right_idx=0;

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

	return result;
}

