/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/CosineDistance.h>
#include <shogun/features/Features.h>

using namespace shogun;

CosineDistance::CosineDistance()
: DenseDistance<float64_t>()
{
}

CosineDistance::CosineDistance(std::shared_ptr<DenseFeatures<float64_t>> l, std::shared_ptr<DenseFeatures<float64_t>> r)
: DenseDistance<float64_t>()
{
	init(l, r);
}

CosineDistance::~CosineDistance()
{
	cleanup();
}

bool CosineDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	return DenseDistance<float64_t>::init(l,r);
}

void CosineDistance::cleanup()
{
}

float64_t CosineDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen)
	float64_t s=0;
	float64_t ab=0;
	float64_t sa=0;
	float64_t sb=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			ab+=avec[i]*bvec[i];
			sa+=pow(fabs(avec[i]),2);
			sb+=pow(fabs(bvec[i]),2);
		}
	}

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	s=sqrt(sa)*sqrt(sb);

	// trap division by zero
	if(s==0)
		return 0;

	s=1-ab/s;
	if(s<0)
		return 0;
	else
		return s ;
}
