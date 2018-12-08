/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Michele Mazzoni, Viktor Gal, 
 *          Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#include <shogun/distance/MahalanobisDistance.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

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
	// FIXME: See comments in
	// https://github.com/shogun-toolbox/shogun/pull/4085#discussion_r166254024
	ASSERT(CRealDistance::init(l, r));

	SGMatrix<float64_t> cov;

	auto feat_l = static_cast<CDenseFeatures<float64_t>*>(l);
	auto feat_r = static_cast<CDenseFeatures<float64_t>*>(r);

	if ( l == r)
	{
		mean = feat_l->get_mean();
		cov = feat_r->get_cov();
	}
	else
	{
		mean = feat_l->compute_mean(feat_l, feat_r);
		cov = CDotFeatures::compute_cov(feat_l, feat_r);
	}

	auto num_features = cov.num_rows;
	chol_cov_L = SGMatrix<float64_t>(num_features, num_features);
	chol_cov_d = SGVector<float64_t>(num_features);
	chol_cov_p = SGVector<index_t>(num_features);
	linalg::ldlt_factor(cov, chol_cov_L, chol_cov_d, chol_cov_p);

	return true;
}

void CMahalanobisDistance::cleanup()
{
}

float64_t CMahalanobisDistance::compute(int32_t idx_a, int32_t idx_b)
{
	auto feat_l = static_cast<CDenseFeatures<float64_t>*>(lhs);
	auto feat_r = static_cast<CDenseFeatures<float64_t>*>(rhs);

	SGVector<float64_t> bvec = feat_r->get_feature_vector(idx_b);

	SGVector<float64_t> diff;
	SGVector<float64_t> avec;

	if (use_mean)
		diff = mean.clone();
	else
	{
		avec = feat_l->get_feature_vector(idx_a);
		diff=avec.clone();
	}

	ASSERT(diff.vlen == bvec.vlen)

	for (int32_t i=0; i < diff.vlen; i++)
		diff[i] = bvec.vector[i] - diff[i];

	auto v = linalg::ldlt_solver(chol_cov_L, chol_cov_d, chol_cov_p, diff);
	auto result = linalg::dot(v, diff);

	if (!use_mean)
		feat_l->free_feature_vector(avec, idx_a);

	feat_r->free_feature_vector(bvec, idx_b);

	if (disable_sqrt)
		return result;
	else
		return std::sqrt(result);
}

void CMahalanobisDistance::init()
{
	disable_sqrt=false;
	use_mean=false;

	SG_ADD(
	    &disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
	SG_ADD(
	    &use_mean, "use_mean", "If distance shall be computed between mean "
	                           "vector and vector from rhs or between lhs and "
	                           "rhs.");
}

