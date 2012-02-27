/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifdef HAVE_LAPACK

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/MahalanobisDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

CMahalanobisDistance::CMahalanobisDistance() : CRealDistance()
{
	init();
}

CMahalanobisDistance::CMahalanobisDistance(CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r)
: CRealDistance()
{
	init();
	init(l, r);
}

CMahalanobisDistance::~CMahalanobisDistance()
{
	cleanup();
}

bool CMahalanobisDistance::init(CFeatures* l, CFeatures* r)
{
	CRealDistance::init(l, r);

	mean = ((CSimpleFeatures<float64_t>*) l)->get_mean();
	icov  = ((CSimpleFeatures<float64_t>*) l)->get_cov();

	CMath::inverse(icov);

	return true;
}

void CMahalanobisDistance::cleanup()
{
}

float64_t CMahalanobisDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t blen;
	bool bfree;
	float64_t* bvec = ((CSimpleFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);

	ASSERT(blen == mean.vlen);

	SGVector<float64_t> diff(bvec, blen);
	for (int32_t i = 0 ; i<blen ; i++)
		diff[i] -= mean[i];

	SGVector<float64_t> v = diff.clone();
	cblas_dgemv(CblasColMajor, CblasNoTrans,
		icov.num_rows, icov.num_cols, 1.0, icov.matrix, 
		diff.vlen, diff.vector, 1, 0.0, v.vector, 1);

	float64_t result = cblas_ddot(v.vlen, v.vector, 1, diff.vector, 1);

	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(bvec, idx_b, bfree);

	if (disable_sqrt)
		return result;
	else
		return CMath::sqrt(result);
}

void CMahalanobisDistance::init()
{
	disable_sqrt = false;

	m_parameters->add(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
}

#endif /* HAVE_LAPACK */
