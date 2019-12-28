/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#include <gtest/gtest.h>
#include <memory>
#include <shogun/distance/LevensteinDistance.h>

using namespace shogun;

TEST(LevensteinDistance, distance)
{
	auto levenstein = std::make_shared<LevensteinDistance>(
	    "GaussianKernels", "GaussianKernel");
	EXPECT_EQ(levenstein->distance(), 1);
	EXPECT_EQ(levenstein->distance("intention", "execution"), 5);
	EXPECT_EQ(levenstein->distance("horse", "ros"), 3);
}