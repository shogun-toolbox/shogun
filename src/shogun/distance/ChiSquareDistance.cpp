/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/ChiSquareDistance.h>
#include <shogun/features/Features.h>

using namespace shogun;

ChiSquareDistance::ChiSquareDistance() : DenseDistance<float64_t>()
{
}

ChiSquareDistance::ChiSquareDistance(std::shared_ptr<DenseFeatures<float64_t>> l, std::shared_ptr<DenseFeatures<float64_t>> r)
: DenseDistance<float64_t>()
{
	init(l, r);
}

ChiSquareDistance::~ChiSquareDistance()
{
	cleanup();
}

bool ChiSquareDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=DenseDistance<float64_t>::init(l,r);

	return result;
}

void ChiSquareDistance::cleanup()
{
}

float64_t ChiSquareDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen)

	float64_t absTmp = 0;
	float64_t result=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			absTmp=fabs(avec[i])+fabs(bvec[i]);
			if(absTmp!=0)
				result+=pow((avec[i]-bvec[i]),2)/absTmp;
		}

	}

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
