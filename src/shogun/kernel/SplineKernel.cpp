/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Bartunov
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/io.h>
#include <shogun/kernel/SplineKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SimpleFeatures.h>

using namespace shogun;

CSplineKernel::CSplineKernel(void)
{
}

CSplineKernel::CSplineKernel(CDotFeatures* l, CDotFeatures* r)
{
	init(l,r);
}

CSplineKernel::~CSplineKernel()
{
	cleanup();
}

bool CSplineKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l,r);
	return init_normalizer();
}

void CSplineKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CSplineKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float32_t* avec = ((CSimpleFeatures<float32_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float32_t* bvec = ((CSimpleFeatures<float32_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen == blen);

	float64_t result = 0;
	for (int32_t i = 0; i < alen; i++) {
		const float64_t x = avec[i], y = bvec[i];
		const float64_t min = CMath::min(avec[i], bvec[i]);
		result += 1 + x*y + x*y*min - ((x+y)/2)*min*min + min*min*min/3;
	}

	((CSimpleFeatures<float32_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float32_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
