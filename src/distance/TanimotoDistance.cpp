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
#include "distance/TanimotoDistance.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CTanimotoDistance::CTanimotoDistance()
: CSimpleDistance<float64_t>()
{
}

CTanimotoDistance::CTanimotoDistance(CRealFeatures* l, CRealFeatures* r)
: CSimpleDistance<float64_t>()
{
	init(l, r);
}

CTanimotoDistance::~CTanimotoDistance()
{
	cleanup();
}

bool CTanimotoDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<float64_t>::init(l,r);

	return result;
}

void CTanimotoDistance::cleanup()
{
}

bool CTanimotoDistance::load_init(FILE* src)
{
	return false;
}

bool CTanimotoDistance::save_init(FILE* dest)
{
	return false;
}

float64_t CTanimotoDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

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
		}
	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	
	s=nx+ny-d;
	
	// trap division by zero
	if(s==0)
		return 0;

	return (d/s);
}
