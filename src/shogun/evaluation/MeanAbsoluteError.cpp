/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2011 Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <evaluation/MeanAbsoluteError.h>
#include <labels/Labels.h>
#include <labels/RegressionLabels.h>
#include <mathematics/Math.h>

using namespace shogun;

float64_t CMeanAbsoluteError::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted && predicted->get_label_type() == LT_REGRESSION)
	ASSERT(ground_truth && ground_truth->get_label_type() == LT_REGRESSION)

	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels())
	int32_t length = predicted->get_num_labels();
	float64_t mae = 0.0;
	for (int32_t i=0; i<length; i++)
		mae += CMath::abs(((CRegressionLabels*) predicted)->get_label(i) - ((CRegressionLabels*) ground_truth)->get_label(i));
	mae /= length;
	return mae;
}
