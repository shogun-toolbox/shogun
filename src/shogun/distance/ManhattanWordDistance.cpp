/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Gehl
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/distance/ManhattanWordDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>

using namespace distance;

CManhattanWordDistance::CManhattanWordDistance()
: CStringDistance<uint16_t>()
{
	SG_DEBUG("CManhattanWordDistance created")
}

CManhattanWordDistance::CManhattanWordDistance(
	CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r)
: CStringDistance<uint16_t>()
{
	SG_DEBUG("CManhattanWordDistance created")

	init(l, r);
}

CManhattanWordDistance::~CManhattanWordDistance()
{
	cleanup();
}

bool CManhattanWordDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CStringDistance<uint16_t>::init(l,r);
	return result;
}

void CManhattanWordDistance::cleanup()
{
}

float64_t CManhattanWordDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=((CStringFeatures<uint16_t>*) lhs)->
		get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=((CStringFeatures<uint16_t>*) rhs)->
		get_feature_vector(idx_b, blen, free_bvec);

	int32_t result=0;

	int32_t left_idx=0;
	int32_t right_idx=0;

	while (left_idx < alen && right_idx < blen)
	{
		uint16_t sym=avec[left_idx];
		if (avec[left_idx]==bvec[right_idx])
		{
			int32_t old_left_idx=left_idx;
			int32_t old_right_idx=right_idx;

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

	((CStringFeatures<uint16_t>*) lhs)->
		free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<uint16_t>*) rhs)->
		free_feature_vector(bvec, idx_b, free_bvec);

	return result;
}

