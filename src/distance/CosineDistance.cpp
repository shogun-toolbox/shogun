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
#include "distance/CosineDistance.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CCosineDistance::CCosineDistance()
: CSimpleDistance<DREAL>()
{
}

CCosineDistance::CCosineDistance(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<DREAL>()
{
	init(l, r);
}

CCosineDistance::~CCosineDistance()
{
	cleanup();
}

bool CCosineDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<DREAL>::init(l,r);

	return result;
}

void CCosineDistance::cleanup()
{
}

bool CCosineDistance::load_init(FILE* src)
{
	return false;
}

bool CCosineDistance::save_init(FILE* dest)
{
	return false;
}

DREAL CCosineDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);
	DREAL s=0;
	DREAL ab=0;
	DREAL sa=0;
	DREAL sb=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			ab+=avec[i]*bvec[i];
			sa+=pow(fabs(avec[i]),2);
			sb+=pow(fabs(bvec[i]),2);
		}
	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	
	s=sqrt(sa)*sqrt(sb);
	
	// trap division by zero
	if(s==0)
		return 0;

	s=1-ab/s;
	if(s<0)
		return 0;
	else
		return s ;
}
