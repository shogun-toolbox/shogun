/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <gtest/gtest.h>
#include <shogun/distance/MahalanobisDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/common.h>

using namespace shogun;

TEST(MahalanobisDistance, compute_distance)
{
	// create four points (0,0) (2,10) (2,8) (5,5)
	SGMatrix<float64_t> rect(2, 4);
	rect(0, 0) = 0;
	rect(1, 0) = 0;
	rect(0, 1) = 2;
	rect(1, 1) = 10;
	rect(0, 2) = 2;
	rect(1, 2) = 8;
	rect(0, 3) = 5;
	rect(1, 3) = 5;

	auto feature = std::make_shared<DenseFeatures<float64_t>>(rect);
	auto distance = std::make_shared<MahalanobisDistance>(feature, feature);
	EXPECT_NEAR(distance->distance(1, 1), 0.0, 1e-10);
	EXPECT_NEAR(distance->distance(1, 3), 2.63447126986, 1e-10);
	EXPECT_NEAR(distance->distance(2, 3), 2.22834405812, 1e-10);
}
