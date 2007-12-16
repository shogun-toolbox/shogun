/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl 
 * Copyright (C) 2006 Fraunhofer Institute FIRST 
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/ChebyshewMetric.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

CChebyshewMetric::CChebyshewMetric()
: CSimpleDistance<DREAL>()
{
}

CChebyshewMetric::CChebyshewMetric(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<DREAL>()
{
	init(l, r);
}

CChebyshewMetric::~CChebyshewMetric()
{
	cleanup();
}

bool CChebyshewMetric::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<DREAL>::init(l,r);

	return result;
}

void CChebyshewMetric::cleanup()
{
}

bool CChebyshewMetric::load_init(FILE* src)
{
	return false;
}

bool CChebyshewMetric::save_init(FILE* dest)
{
	return false;
}

DREAL CChebyshewMetric::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	DREAL result=DBL_MIN;

	for (INT i=0; i<alen; i++)
		result=CMath::max(result, fabs(avec[i]-bvec[i]));

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
