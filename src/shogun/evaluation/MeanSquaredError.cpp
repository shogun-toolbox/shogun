/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Fernando Iglesias
 */

#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

float64_t MeanSquaredError::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	REQUIRE(predicted, "Predicted labels must be not null.\n")
	REQUIRE(ground_truth, "Ground truth labels must be not null.\n")
	REQUIRE(predicted->get_num_labels() == ground_truth->get_num_labels(), "The number of predicted labels (%d) must be equal to the number of ground truth labels (%d).\n")
	int32_t length = predicted->get_num_labels();
	float64_t mse = 0.0;
	auto predicted_regression = regression_labels(predicted);
	auto ground_truth_regression = regression_labels(ground_truth);

	for (int32_t i=0; i<length; i++)
		mse += Math::sq(
		    predicted_regression->get_label(i) -
		    ground_truth_regression->get_label(i));
	mse /= length;
	return mse;
}
