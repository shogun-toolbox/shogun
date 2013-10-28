/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;

CEuclideanDistance::CEuclideanDistance() : CRealDistance()
{
	init();
}

CEuclideanDistance::CEuclideanDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
: CRealDistance()
{
	init();
	init(l, r);
}

CEuclideanDistance::~CEuclideanDistance()
{
	cleanup();
}

bool CEuclideanDistance::init(CFeatures* l, CFeatures* r)
{
	CRealDistance::init(l, r);

	return true;
}

void CEuclideanDistance::cleanup()
{
}

float64_t CEuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t result=0;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
		get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	for (int32_t i=0; i<alen; i++)
		result+=CMath::sq(avec[i] - bvec[i]);

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	if (disable_sqrt)
		return result;

	return CMath::sqrt(result);
}

void CEuclideanDistance::init()
{
	disable_sqrt=false;

	m_parameters->add(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
}

float64_t CEuclideanDistance::distance_upper_bounded(int32_t idx_a, int32_t idx_b, float64_t upper_bound)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t result=0;

	upper_bound *= upper_bound;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
		get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	for (int32_t i=0; i<alen; i++)
	{
		result+=CMath::sq(avec[i] - bvec[i]);

		if (result > upper_bound)
		{
			((CDenseFeatures<float64_t>*) lhs)->
				free_feature_vector(avec, idx_a, afree);
			((CDenseFeatures<float64_t>*) rhs)->
				free_feature_vector(bvec, idx_b, bfree);

			if (disable_sqrt)
				return result;
			else
				return CMath::sqrt(result);
		}
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	if (disable_sqrt)
		return result;
	else
		return CMath::sqrt(result);
}
