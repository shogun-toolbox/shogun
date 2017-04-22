/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

float64_t CMeanSquaredError::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	REQUIRE(predicted, "Predicted labels must be not null.\n")
	REQUIRE(ground_truth, "Ground truth labels must be not null.\n")
	REQUIRE(predicted->get_num_labels() == ground_truth->get_num_labels(), "The number of predicted labels (%d) must be equal to the number of ground truth labels (%d).\n")
	REQUIRE(predicted->get_label_type() == LT_REGRESSION, "Predicted label type (%d) must be regression (%d).\n", predicted->get_label_type(), LT_REGRESSION)
	REQUIRE(ground_truth->get_label_type() == LT_REGRESSION, "Ground truth label type (%d) must be regression (%d).\n", ground_truth->get_label_type(), LT_REGRESSION)
	int32_t length = predicted->get_num_labels();
	float64_t mse = 0.0;
	for (int32_t i=0; i<length; i++)
		mse += CMath::sq(((CRegressionLabels*) predicted)->get_label(i) - ((CRegressionLabels*) ground_truth)->get_label(i));
	mse /= length;
	return mse;
}
