/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/JensenMetric.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CJensenMetric::CJensenMetric()
: CSimpleDistance<float64_t>()
{
}

CJensenMetric::CJensenMetric(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<float64_t>()
{
	init(l, r);
}

CJensenMetric::~CJensenMetric()
{
	cleanup();
}

bool CJensenMetric::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<float64_t>::init(l,r);

	return result;
}

void CJensenMetric::cleanup()
{
}

bool CJensenMetric::load_init(FILE* src)
{
	return false;
}

bool CJensenMetric::save_init(FILE* dest)
{
	return false;
}

float64_t CJensenMetric::compute(int32_t idx_a, int32_t idx_b)
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
			absTmp=0.5*(avec[i]+bvec[i]);
			if(avec[i]>0)
				result+=avec[i]*log(avec[i]/absTmp);
			if(bvec[i]>0)
				result+=bvec[i]*log(bvec[i]/absTmp);
		}

	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);


	return result;
}
