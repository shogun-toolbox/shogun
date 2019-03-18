/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#include <gtest/gtest.h>
#include <shogun/util/utils.h>
#include <limits>

using namespace shogun;

TEST(utils, correct_cast)
{
	size_t input = 10;
	EXPECT_NO_THROW(size_t_to_int32_cast(input));
}

TEST(utils, wrong_cast_overflow)
{
	size_t input = std::numeric_limits<int32_t>::max()+1;
	EXPECT_THROW(size_t_to_int32_cast(input), std::overflow_error);
}