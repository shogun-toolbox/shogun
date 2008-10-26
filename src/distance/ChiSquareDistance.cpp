/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Christian Gehl
 * Copyright (C) 2008 Fraunhofer Institute FIRST
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/ChiSquareDistance.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CChiSquareDistance::CChiSquareDistance()
: CSimpleDistance<DREAL>()
{
}

CChiSquareDistance::CChiSquareDistance(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<DREAL>()
{
	init(l, r);
}

CChiSquareDistance::~CChiSquareDistance()
{
	cleanup();
}

bool CChiSquareDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<DREAL>::init(l,r);

	return result;
}

void CChiSquareDistance::cleanup()
{
}

bool CChiSquareDistance::load_init(FILE* src)
{
	return false;
}

bool CChiSquareDistance::save_init(FILE* dest)
{
	return false;
}

DREAL CChiSquareDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	DREAL absTmp = 0;
	DREAL result=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			absTmp=fabs(avec[i])+fabs(bvec[i]);
			if(absTmp!=0)
				result+=pow((avec[i]-bvec[i]),2)/absTmp;
		}

	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	
	return result;
}
