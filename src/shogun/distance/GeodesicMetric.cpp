/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/GeodesicMetric.h>
#include <shogun/features/Features.h>

using namespace distance;

CGeodesicMetric::CGeodesicMetric() : CDenseDistance<float64_t>()
{
}

CGeodesicMetric::CGeodesicMetric(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
: CDenseDistance<float64_t>()
{
	init(l, r);
}

CGeodesicMetric::~CGeodesicMetric()
{
	cleanup();
}

bool CGeodesicMetric::init(CFeatures* l, CFeatures* r)
{
	bool result=CDenseDistance<float64_t>::init(l,r);

	return result;
}

void CGeodesicMetric::cleanup()
{
}

float64_t CGeodesicMetric::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen)

	float64_t s=0;
	float64_t d=0;
	float64_t nx=0;
	float64_t ny=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			d+=avec[i]*bvec[i];
			nx+=avec[i]*avec[i];
			ny+=bvec[i]*bvec[i];
			s+=avec[i]+bvec[i];
		}
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);


	// trap division by zero
	if(s==0 || nx==0 || ny==0)
		return 0;

	d/=CMath::sqrt(nx*ny);

	// can only happen due to numerical problems
	if (CMath::abs(d)>1.0)
		d=CMath::sign(d);

	return acos(d);
}
