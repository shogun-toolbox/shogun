/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/SparseEuclideanDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/SparseFeatures.h>

using namespace shogun;

SparseEuclideanDistance::SparseEuclideanDistance()
: SparseDistance<float64_t>()
{
	init();
}

SparseEuclideanDistance::SparseEuclideanDistance(
	const std::shared_ptr<SparseFeatures<float64_t>>& l, const std::shared_ptr<SparseFeatures<float64_t>>& r)
: SparseDistance<float64_t>()
{
	init();
	init(l, r);
}

SparseEuclideanDistance::~SparseEuclideanDistance()
{
	cleanup();
}

bool SparseEuclideanDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	SparseDistance<float64_t>::init(l, r);

	cleanup();

	sq_lhs=SG_MALLOC(float64_t, lhs->get_num_vectors());
	sq_lhs=(std::static_pointer_cast<SparseFeatures<float64_t>>(lhs))->compute_squared(sq_lhs);

	if (lhs==rhs)
		sq_rhs=sq_lhs;
	else
	{
		sq_rhs=SG_MALLOC(float64_t, rhs->get_num_vectors());
		sq_rhs=(std::static_pointer_cast<SparseFeatures<float64_t>>(rhs))->compute_squared(sq_rhs);
	}

	return true;
}

void SparseEuclideanDistance::cleanup()
{
	if (sq_lhs != sq_rhs)
		SG_FREE(sq_rhs);
	sq_rhs = NULL;

	SG_FREE(sq_lhs);
	sq_lhs = NULL;
}

float64_t SparseEuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=(std::static_pointer_cast<SparseFeatures<float64_t>>(lhs))->compute_squared_norm(
		std::static_pointer_cast<SparseFeatures<float64_t>>(lhs), sq_lhs, idx_a,
		std::static_pointer_cast<SparseFeatures<float64_t>>(rhs), sq_rhs, idx_b);

	return std::sqrt(result);
}

void SparseEuclideanDistance::init()
{
	sq_lhs=NULL;
	sq_rhs=NULL;
}
