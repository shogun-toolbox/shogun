/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/TanimotoDistance.h>
#include <shogun/features/Features.h>

using namespace shogun;

TanimotoDistance::TanimotoDistance()
: DenseDistance<float64_t>()
{
}

TanimotoDistance::TanimotoDistance(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r)
: DenseDistance<float64_t>()
{
	init(l, r);
}

TanimotoDistance::~TanimotoDistance()
{
	cleanup();
}

bool TanimotoDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=DenseDistance<float64_t>::init(l,r);

	return result;
}

void TanimotoDistance::cleanup()
{
}

float64_t TanimotoDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen)

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

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	s=nx+ny-d;

	// trap division by zero
	if(s==0)
		return 0;

	return (d/s);
}
