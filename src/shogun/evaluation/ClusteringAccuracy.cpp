/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang
 */

#include <shogun/evaluation/ClusteringAccuracy.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

float64_t ClusteringAccuracy::evaluate_impl(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	ASSERT(predicted && ground_truth)
	ASSERT(ground_truth->get_label_type() == LT_MULTICLASS)
	ASSERT(predicted->get_label_type() == LT_MULTICLASS)
	SGVector<int32_t> predicted_ilabels=multiclass_labels(predicted)->get_int_labels();
	SGVector<int32_t> groundtruth_ilabels=multiclass_labels(ground_truth)->get_int_labels();
	int32_t correct=0;
	for (int32_t i=0; i < predicted_ilabels.vlen; ++i)
	{
		if (predicted_ilabels[i] == groundtruth_ilabels[i])
			correct++;
	}
	return float64_t(correct)/predicted_ilabels.vlen;
}
