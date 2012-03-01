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

	if ( ((CSimpleFeatures<float64_t>*) l)->is_equal((CSimpleFeatures<float64_t>*) r) )
	{
		icov  = ((CSimpleFeatures<float64_t>*) l)->get_cov();
		equal_features = true;
	}
	else
	{
		icov = CDotFeatures::compute_cov((CDotFeatures*)lhs, (CDotFeatures*)rhs);
	}

	CMath::inverse(icov);

	return true;
}

void CMahalanobisDistance::cleanup()
{
}

float64_t CMahalanobisDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t* avec;

	float64_t* bvec = ((CSimpleFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);

	SGVector<float64_t> c;

	if (equal_features)
	{
		c = mean.clone();
	}
	else
	{
		avec = ((CSimpleFeatures<float64_t>*) lhs)-> 
			get_feature_vector(idx_a, alen, afree);

		c.resize_vector(alen);
		for (int i = 0; i < alen; i++)
			c[i] = avec[i];

		((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	}

	ASSERT(blen == c.vlen);

	SGVector<float64_t> diff;
	diff.resize_vector(blen);
	for (int32_t i = 0 ; i < diff.vlen ; i++)
		diff[i] = bvec[i] - c[i];

	SGVector<float64_t> v = diff.clone();
	cblas_dgemv(CblasColMajor, CblasNoTrans,
		icov.num_rows, icov.num_cols, 1.0, icov.matrix, 
		diff.vlen, diff.vector, 1, 0.0, v.vector, 1);

	float64_t result = cblas_ddot(v.vlen, v.vector, 1, diff.vector, 1);

	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	v.destroy_vector();
	c.destroy_vector();

	if (disable_sqrt)
		return result;
	else
		return CMath::sqrt(result);
}

void CMahalanobisDistance::init()
{
	disable_sqrt   = false;
	equal_features = false;

	m_parameters->add(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
}

#endif /* HAVE_LAPACK */
