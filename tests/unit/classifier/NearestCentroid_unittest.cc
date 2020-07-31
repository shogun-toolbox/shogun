/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include <gtest/gtest.h>
#include <shogun/classifier/NearestCentroid.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;
TEST(NearestCentroid, fit_and_predict)
{
	SGMatrix<float64_t> X{{-10, -1}, {-2, -1}, {-3, -2},
	                      {1, 1},    {2, 1},   {3, 2}};
	SGVector<float64_t> y{0, 0, 0, 1, 1, 1};

	auto train_data = std::make_shared<DenseFeatures<float64_t>>(X);
	auto train_labels = std::make_shared<MulticlassLabels>(y);
	auto distance = std::make_shared<EuclideanDistance>();

	SGMatrix<float64_t> t{{3, 2}, {-10, -1}, {-100, 100}};
	auto test_data = std::make_shared<DenseFeatures<float64_t>>(t);
	auto clf = std::make_shared<NearestCentroid>(distance);
	clf->train(train_data, train_labels);
	auto result_labels = clf->apply(test_data);
	auto result = result_labels->as<MulticlassLabels>()->get_labels();
	EXPECT_EQ(result[0], 1);
	EXPECT_EQ(result[1], 0);
	EXPECT_EQ(result[2], 0);
}