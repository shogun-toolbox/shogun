/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Miguel Angel Bautista
 * Copyright (C) 2011 Berlin Institute of Technology and Max Planck Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/AttenuatedEuclideanDistance.h>
#include <shogun/features/Features.h>

using namespace distance;

CAttenuatedEuclideanDistance::CAttenuatedEuclideanDistance() : CRealDistance()
{
	init();
}

CAttenuatedEuclideanDistance::CAttenuatedEuclideanDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
: CRealDistance()
{
	init();
	init(l, r);
}

CAttenuatedEuclideanDistance::~CAttenuatedEuclideanDistance()
{
	cleanup();
}

bool CAttenuatedEuclideanDistance::init(CFeatures* l, CFeatures* r)
{
	CRealDistance::init(l, r);
	return true;
}

void CAttenuatedEuclideanDistance::cleanup()
{
}

float64_t CAttenuatedEuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t result=0;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
		get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	for (int32_t i=0; i<alen; i++)
		result+=(CMath::abs(avec[i])*CMath::abs(bvec[i]))*CMath::pow(avec[i] - bvec[i],2);

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	if (disable_sqrt)
		return result;

	return CMath::sqrt(result);
}

void CAttenuatedEuclideanDistance::init()
{
	disable_sqrt=false;

	m_parameters->add(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
}
