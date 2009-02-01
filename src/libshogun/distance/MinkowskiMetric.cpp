/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/MinkowskiMetric.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CMinkowskiMetric::CMinkowskiMetric(float64_t k_)
: CSimpleDistance<float64_t>(), k(k_)
{
}

CMinkowskiMetric::CMinkowskiMetric(
	CRealFeatures* l, CRealFeatures* r, float64_t k_)
: CSimpleDistance<float64_t>(), k(k_)
{
	init(l, r);
}

CMinkowskiMetric::~CMinkowskiMetric()
{
	cleanup();
}

bool CMinkowskiMetric::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<float64_t>::init(l,r);

	return result;
}

void CMinkowskiMetric::cleanup()
{
}

bool CMinkowskiMetric::load_init(FILE* src)
{
	return false;
}

bool CMinkowskiMetric::save_init(FILE* dest)
{
	return false;
}

float64_t CMinkowskiMetric::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	float64_t absTmp = 0;
	float64_t result=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			absTmp=fabs(avec[i]-bvec[i]);
			result+=pow(absTmp,k);
		}

	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return pow(result,1/k);
}
