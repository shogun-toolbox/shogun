/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/base/range.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>


using namespace shogun;

class RegressionLabels : public ::testing::Test
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

TEST_F(RegressionLabels, regression_labels_from_regression)
{
	auto labels = some<CRegressionLabels>(labels_regression);
	auto labels2 = regression_labels(labels);
	EXPECT_EQ(labels, labels2);
}

TEST_F(RegressionLabels, regression_labels_from_binary)
{
	auto labels = some<CBinaryLabels>(n);
	labels->set_labels(labels_regression);
	labels->set_values(scores);

	auto labels2 = regression_labels(labels);
	EXPECT_NE(labels, labels2);
	ASSERT_NE(labels2.get(), nullptr);
	EXPECT_EQ(labels->get_labels(), labels2->get_labels());
	EXPECT_EQ(labels->get_values(), labels2->get_values());
}
