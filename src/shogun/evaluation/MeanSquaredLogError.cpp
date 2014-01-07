/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <evaluation/MeanSquaredLogError.h>
#include <labels/Labels.h>
#include <labels/RegressionLabels.h>
#include <mathematics/Math.h>

using namespace shogun;

float64_t CMeanSquaredLogError::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted && ground_truth)
	ASSERT(predicted->get_num_labels()==ground_truth->get_num_labels())
	ASSERT(predicted->get_label_type()==LT_REGRESSION)
	ASSERT(ground_truth->get_label_type()==LT_REGRESSION)

	int32_t length=predicted->get_num_labels();
	float64_t msle=0.0;
	for (int32_t i=0; i<length; i++)
	{
		float64_t prediction=((CRegressionLabels*) predicted)->get_label(i);
		float64_t truth=((CRegressionLabels*) ground_truth)->get_label(i);

		if (prediction<=-1.0 || truth<=-1.0)
		{
			SG_WARNING("Negative label[%d] in %s is not allowed, ignoring!\n",
					i, get_name());
			continue;
		}

		float64_t a=CMath::log(prediction+1);
		float64_t b=CMath::log(truth+1);
		msle+=CMath::sq(a-b);
	}
	msle /= length;
	return CMath::sqrt(msle);
}
