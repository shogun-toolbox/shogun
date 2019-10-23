/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/distance/HammingWordDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

HammingWordDistance::HammingWordDistance()
{
	init();
}

HammingWordDistance::HammingWordDistance(bool sign)
: StringDistance<uint16_t>()
{
	init();
	use_sign=sign;

	SG_DEBUG("CHammingWordDistance with sign: {} created", (sign) ? 1 : 0)
}

HammingWordDistance::HammingWordDistance(
	const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r, bool sign)
: StringDistance<uint16_t>()
{
	init();
	use_sign=sign;

	SG_DEBUG("CHammingWordDistance with sign: {} created", (sign) ? 1 : 0)

	init(l, r);
}

HammingWordDistance::~HammingWordDistance()
{
	cleanup();
}

bool HammingWordDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=StringDistance<uint16_t>::init(l,r);
	return result;
}

void HammingWordDistance::cleanup()
{
}

float64_t HammingWordDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs))->
		get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=(std::static_pointer_cast<StringFeatures<uint16_t>>(rhs))->
		get_feature_vector(idx_b, blen, free_bvec);

	int32_t result=0;

	int32_t left_idx=0;
	int32_t right_idx=0;

	if (use_sign)
	{
		// hamming of: if words appear in both vectors
		while (left_idx < alen && right_idx < blen)
		{
			uint16_t sym=avec[left_idx];
			if (avec[left_idx]==bvec[right_idx])
			{
				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;
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
	}
	else
	{
		//hamming of: if words appear in both vectors _the same number_ of times
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

				if ((left_idx-old_left_idx)!=(right_idx-old_right_idx))
					result++;
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

void HammingWordDistance::init()
{
	use_sign = false;
	SG_ADD(
	    &use_sign, "use_sign", "If signum(counts) is used instead of counts.");
}
