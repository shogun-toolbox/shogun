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
: CSimpleDistance<DREAL>()
{
}

CJensenMetric::CJensenMetric(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<DREAL>()
{
	init(l, r);
}

CJensenMetric::~CJensenMetric()
{
	cleanup();
}

bool CJensenMetric::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<DREAL>::init(l,r);

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

DREAL CJensenMetric::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	DREAL absTmp = 0;
	DREAL result=0;
	{
		for (INT i=0; i<alen; i++)
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
