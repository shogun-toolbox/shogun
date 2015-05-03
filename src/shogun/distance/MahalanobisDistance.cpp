/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/MahalanobisDistance.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

CMahalanobisDistance::CMahalanobisDistance() : CRealDistance()
{
	init();
}

CMahalanobisDistance::CMahalanobisDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
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


	if ( l == r)
	{
		mean = ((CDenseFeatures<float64_t>*) l)->get_mean();
		icov  = ((CDenseFeatures<float64_t>*) l)->get_cov();
	}
	else
	{
		mean = ((CDenseFeatures<float64_t>*) l)->get_mean((CDotFeatures*) lhs, (CDotFeatures*) rhs);
		icov = CDotFeatures::compute_cov((CDotFeatures*) lhs, (CDotFeatures*) rhs);
	}

	SGMatrix<float64_t>::inverse(icov);

	return true;
}

void CMahalanobisDistance::cleanup()
{
}

float64_t CMahalanobisDistance::compute(int32_t idx_a, int32_t idx_b)
{

	SGVector<float64_t> bvec = ((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b);

	SGVector<float64_t> diff;
	SGVector<float64_t> avec;

	if (use_mean)
		diff = mean.clone();
	else
	{
		avec = ((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a);
		diff=avec.clone();
	}

	ASSERT(diff.vlen == bvec.vlen)

	for (int32_t i=0; i < diff.vlen; i++)
		diff[i] = bvec.vector[i] - diff[i];

	SGVector<float64_t> v = diff.clone();
	cblas_dgemv(CblasColMajor, CblasNoTrans,
		icov.num_rows, icov.num_cols, 1.0, icov.matrix,
		diff.vlen, diff.vector, 1, 0.0, v.vector, 1);

	float64_t result = cblas_ddot(v.vlen, v.vector, 1, diff.vector, 1);

	if (!use_mean)
		((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a);

	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b);

	if (disable_sqrt)
		return result;
	else
		return CMath::sqrt(result);
}

void CMahalanobisDistance::init()
{
	disable_sqrt=false;
	use_mean=false;

	m_parameters->add(&disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
	m_parameters->add(&use_mean, "use_mean", "If distance shall be computed between mean vector and vector from rhs or between lhs and rhs.");
}

#endif /* HAVE_LAPACK */
