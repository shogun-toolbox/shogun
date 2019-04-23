/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/distance/CanberraWordDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CanberraWordDistance::CanberraWordDistance()
: StringDistance<uint16_t>()
{
	SG_DEBUG("CCanberraWordDistance created")
}

CanberraWordDistance::CanberraWordDistance(
	std::shared_ptr<StringFeatures<uint16_t>> l, std::shared_ptr<StringFeatures<uint16_t>> r)
: StringDistance<uint16_t>()
{
	SG_DEBUG("CCanberraWordDistance created")

	init(l, r);
}

CanberraWordDistance::~CanberraWordDistance()
{
	cleanup();
}

bool CanberraWordDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	return StringDistance<uint16_t>::init(l,r);
}

void CanberraWordDistance::cleanup()
{
}

float64_t CanberraWordDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs))->
		get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=(std::static_pointer_cast<StringFeatures<uint16_t>>(rhs))->
		get_feature_vector(idx_b, blen, free_bvec);

	float64_t result=0;

	int32_t left_idx=0;
	int32_t right_idx=0;

	while (left_idx < alen && right_idx < blen)
	{
		uint16_t sym=avec[left_idx];
		if (avec[left_idx]==bvec[right_idx])
		{
			int32_t old_left_idx=left_idx;
			int32_t old_right_idx=right_idx;

			while (left_idx< alen && avec[left_idx]==sym)
				left_idx++;

			while (right_idx< blen && bvec[right_idx]==sym)
				right_idx++;

			result +=
				Math::abs((float64_t)
					((left_idx-old_left_idx)-(right_idx-old_right_idx)))/
				((float64_t)
					((left_idx-old_left_idx) + (right_idx-old_right_idx)));
		}
		else if (avec[left_idx]<bvec[right_idx])
		{
			result++;

			while (left_idx< alen && avec[left_idx]==sym)
				left_idx++;
		}
		else
		{
			sym=bvec[right_idx];
			result++;

			while (right_idx< blen && bvec[right_idx]==sym)
				right_idx++;
		}
	}

	while (left_idx < alen)
	{
		uint16_t sym=avec[left_idx];
		result++;

		while (left_idx< alen && avec[left_idx]==sym)
			left_idx++;
	}

	while (right_idx < blen)
	{
		uint16_t sym=bvec[right_idx];
		result++;

		while (right_idx< blen && bvec[right_idx]==sym)
			right_idx++;
	}
	(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs))->
		free_feature_vector(avec, idx_a, free_avec);
	(std::static_pointer_cast<StringFeatures<uint16_t>>(rhs))->
		free_feature_vector(bvec, idx_b, free_bvec);

	return result;
}
