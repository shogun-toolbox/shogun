/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/evaluation/MeanSquaredLogError.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>

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

		float64_t a = std::log(prediction + 1);
		float64_t b = std::log(truth + 1);
		msle+=CMath::sq(a-b);
	}
	msle /= length;
	return CMath::sqrt(msle);
}
