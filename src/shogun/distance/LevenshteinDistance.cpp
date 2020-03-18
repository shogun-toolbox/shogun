
/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#include <algorithm>
#include <shogun/distance/LevenshteinDistance.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

LevenshteinDistance::LevenshteinDistance() : Distance()
{
}

LevenshteinDistance::LevenshteinDistance(
    std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	init(l, r);
}

bool LevenshteinDistance::init(
    std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Distance::init(l, r);
	auto casted_lhs = std::dynamic_pointer_cast<StringFeatures<char>>(lhs);
	auto casted_rhs = std::dynamic_pointer_cast<StringFeatures<char>>(rhs);

	require(
	    casted_lhs != nullptr,
	    "Left hand side feature must be StringFeatures<char>!");
	require(
	    casted_rhs != nullptr,
	    "Right hand side feature must be StringFeatures<char>!");
	return true;
}

std::shared_ptr<Features>
LevenshteinDistance::replace_rhs(std::shared_ptr<Features> r)
{
	auto previous_rhs = Distance::replace_rhs(r);
	return previous_rhs;
}

std::shared_ptr<Features>
LevenshteinDistance::replace_lhs(std::shared_ptr<Features> l)
{
	auto previous_lhs = Distance::replace_lhs(l);
	return previous_lhs;
}

float64_t LevenshteinDistance::compute(int32_t idx_a, int32_t idx_b)
{
	auto casted_lhs = std::dynamic_pointer_cast<StringFeatures<char>>(lhs);
	auto casted_rhs = std::dynamic_pointer_cast<StringFeatures<char>>(rhs);

	SGVector<char> lhs_str = casted_lhs->get_feature_vector(idx_a);
	SGVector<char> rhs_str = casted_rhs->get_feature_vector(idx_b);
	return compute_impl(lhs_str, rhs_str);
}

float64_t LevenshteinDistance::compute_impl(
    const SGVector<char>& lhs_str, const SGVector<char>& rhs_str)
{
	int lhs_size = lhs_str.vlen, rhs_size = rhs_str.vlen;
	SGMatrix<int> dist(lhs_size + 1, rhs_size + 1);
	for (int i = 0; i <= lhs_size; ++i)
		dist(i, 0) = i;
	for (int i = 0; i <= rhs_size; ++i)
		dist(0, i) = i;
	for (int i = 1; i <= lhs_size; ++i)
	{
		for (int j = 1; j <= rhs_size; ++j)
		{
			if (lhs_str[i - 1] == rhs_str[j - 1])
			{
				dist(i, j) = dist(i - 1, j - 1);
			}
			else
			{
				dist(i, j) = std::min(
				                 dist(i - 1, j - 1),
				                 std::min(dist(i - 1, j), dist(i, j - 1))) +
				             1;
			}
		}
	}
	return dist(lhs_size, rhs_size);
}