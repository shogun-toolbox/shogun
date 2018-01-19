/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/evaluation/MeanAbsoluteError.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>

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
