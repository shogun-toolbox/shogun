/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#include "sg_gtest_utilities.h"
#include <cmath>
#include <memory>
#include <shogun/evaluation/MeanAbsoluteError.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/evaluation/MeanSquaredLogError.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>

class MeanSquaredTest : public ::testing::Test
{
public:
	std::shared_ptr<BinaryLabels> predicted_labels;
	std::shared_ptr<BinaryLabels> ground_truth_labels;
	std::shared_ptr<RegressionLabels> prediect_reg_labels;
	std::shared_ptr<RegressionLabels> gtruth_reg_labels;

protected:
	void SetUp() override
	{
		float64_t threshold = 0.5;
		SGVector<float64_t> prediected_prob = {0.1, 0.4, 0.6, 0.9};
		SGVector<float64_t> gtruth_prob = {0.1, 0.4, 0.4, 0.9};
		predicted_labels =
		    std::make_shared<BinaryLabels>(prediected_prob, threshold);
		ground_truth_labels =
		    std::make_shared<BinaryLabels>(gtruth_prob, threshold);
		auto e = std::exp(1);
		SGVector<float64_t> prediected_reg = {0, e - 1, 0, e - 1};
		SGVector<float64_t> gtruth_reg = {e - 1, 0, e - 1, 0};
		prediect_reg_labels =
		    std::make_shared<RegressionLabels>(prediected_reg);
		gtruth_reg_labels = std::make_shared<RegressionLabels>(gtruth_reg);
	}
};

TEST_F(MeanSquaredTest, MeanSquaredError)
{
	auto MSE = std::make_shared<MeanSquaredError>();
	EXPECT_NEAR(MSE->evaluate(predicted_labels, predicted_labels), 0.0, 1e-11);
	EXPECT_NEAR(
	    MSE->evaluate(predicted_labels, ground_truth_labels), 1.0, 1e-11);
}

TEST_F(MeanSquaredTest, MeanAbsoluteError)
{
	auto MSE = std::make_shared<MeanAbsoluteError>();
	EXPECT_NEAR(MSE->evaluate(predicted_labels, predicted_labels), 0.0, 1e-11);
	EXPECT_NEAR(
	    MSE->evaluate(predicted_labels, ground_truth_labels), 0.5, 1e-11);
}

TEST_F(MeanSquaredTest, MeanSquaredLogError)
{
	auto MSE = std::make_shared<MeanSquaredLogError>();
	EXPECT_NEAR(
	    MSE->evaluate(prediect_reg_labels, gtruth_reg_labels), 1.0, 1e-11);
	EXPECT_NEAR(
	    MSE->evaluate(prediect_reg_labels, prediect_reg_labels), 0.0, 1e-11);
}