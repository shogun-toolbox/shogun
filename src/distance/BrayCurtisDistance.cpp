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
#include "distance/BrayCurtisDistance.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CBrayCurtisDistance::CBrayCurtisDistance()
: CSimpleDistance<DREAL>()
{
}

CBrayCurtisDistance::CBrayCurtisDistance(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<DREAL>()
{
	init(l, r);
}

CBrayCurtisDistance::~CBrayCurtisDistance()
{
	cleanup();
}

bool CBrayCurtisDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<DREAL>::init(l,r);

	return result;
}

void CBrayCurtisDistance::cleanup()
{
}

bool CBrayCurtisDistance::load_init(FILE* src)
{
	return false;
}

bool CBrayCurtisDistance::save_init(FILE* dest)
{
	return false;
}

DREAL CBrayCurtisDistance::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	DREAL s1=0;
	DREAL s2=0;
	{
		for (INT i=0; i<alen; i++)
		{
			s1+=fabs(avec[i]-bvec[i]);
			s2+=fabs(avec[i]+bvec[i]);
		}
	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	
	// trap division by zero
	if(s2==0)
		return 0;

	return (s1/s2);
}
