/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#include <gtest/gtest.h>
#include <memory>
#include <shogun/distance/LevenshteinDistance.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

std::shared_ptr<StringFeatures<char>> create_string_lhs()
{
	std::vector<SGVector<char>> strings = {
	    {'i', 'n', 't', 'e', 'n', 't', 'i', 'o', 'n'},
	    {'h', 'o', 'r', 's', 'e'},
	    {'G', 'a', 'u', 's', 's', 'i', 'a', 'n', 'K', 'e', 'r', 'n', 'e', 'l',
	     's'}};
	return std::make_shared<StringFeatures<char>>(strings, RAWBYTE);
}

std::shared_ptr<StringFeatures<char>> create_string_rhs()
{
	std::vector<SGVector<char>> strings = {
	    {'e', 'x', 'e', 'c', 'u', 't', 'i', 'o', 'n'},
	    {'r', 'o', 's'},
	    {'G', 'a', 'u', 's', 's', 'i', 'a', 'n', 'K', 'e', 'r', 'n', 'e', 'l'}};
	return std::make_shared<StringFeatures<char>>(strings, RAWBYTE);
}

TEST(LevenshteinDistance, distance)
{
	auto features_lhs = create_string_lhs();
	auto features_rhs = create_string_rhs();
	auto levenshtein =
	    std::make_shared<LevenshteinDistance>(features_lhs, features_rhs);

	EXPECT_EQ(levenshtein->distance(0, 0), 5);
	EXPECT_EQ(levenshtein->distance(1, 1), 3);
	EXPECT_EQ(levenshtein->distance(2, 2), 1);
}
