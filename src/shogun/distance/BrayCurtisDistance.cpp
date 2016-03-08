/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Christian Gehl
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/BrayCurtisDistance.h>
#include <shogun/features/Features.h>

using namespace distance;

CBrayCurtisDistance::CBrayCurtisDistance()
: CDenseDistance<float64_t>()
{
}

CBrayCurtisDistance::CBrayCurtisDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
: CDenseDistance<float64_t>()
{
	init(l, r);
}

CBrayCurtisDistance::~CBrayCurtisDistance()
{
	cleanup();
}

bool CBrayCurtisDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CDenseDistance<float64_t>::init(l,r);

	return result;
}

void CBrayCurtisDistance::cleanup()
{
}

float64_t CBrayCurtisDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen)

	float64_t s1=0;
	float64_t s2=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			s1+=fabs(avec[i]-bvec[i]);
			s2+=fabs(avec[i]+bvec[i]);
		}
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	// trap division by zero
	if(s2==0)
		return 0;

	return (s1/s2);
}
