/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/base/range.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>

using namespace shogun;

class RegressionLabelsTest : public ::testing::Test
{
public:
	SGVector<float64_t> labels_regression;
	SGVector<float64_t> scores;

	const int32_t n = 4;

	virtual void SetUp()
	{
		labels_regression = {1, -1.5, 20.6, 0.9};
		scores = {2, 3, 4, 5};
	}

	virtual void TearDown()
	{
	}
};

TEST_F(RegressionLabelsTest, regression_labels_from_regression)
{
	auto labels = std::make_shared<RegressionLabels>(labels_regression);
	auto labels2 = regression_labels(labels);
	EXPECT_EQ(labels, labels2);
}
