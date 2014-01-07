/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Koen van de Sande
 * Copyright (C) 2010 Koen van de Sande / University of Amsterdam
 */

#include <lib/common.h>
#include <kernel/HistogramIntersectionKernel.h>
#include <features/Features.h>
#include <io/SGIO.h>

using namespace shogun;

CHistogramIntersectionKernel::CHistogramIntersectionKernel()
: CDotKernel(0), m_beta(1.0)
{
	register_params();
}

CHistogramIntersectionKernel::CHistogramIntersectionKernel(int32_t size)
: CDotKernel(size), m_beta(1.0)
{
	register_params();
}

CHistogramIntersectionKernel::CHistogramIntersectionKernel(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
	float64_t beta, int32_t size)
: CDotKernel(size), m_beta(beta)
{
	init(l,r);
	register_params();
}

CHistogramIntersectionKernel::~CHistogramIntersectionKernel()
{
	cleanup();
}

bool CHistogramIntersectionKernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CDotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t CHistogramIntersectionKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;

	// checking if beta is default or not
	if (m_beta == 1.0)
	{
		// compute standard histogram intersection kernel
		for (int32_t i=0; i<alen; i++)
			result += (avec[i] < bvec[i]) ? avec[i] : bvec[i];
	}
	else
	{
		//compute generalized histogram intersection kernel
		for (int32_t i=0; i<alen; i++)
			result += CMath::min(CMath::pow(avec[i],m_beta), CMath::pow(bvec[i],m_beta));
	}
	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void CHistogramIntersectionKernel::register_params()
{
	SG_ADD(&m_beta, "beta", "the beta parameter of the kernel", MS_AVAILABLE);
}
