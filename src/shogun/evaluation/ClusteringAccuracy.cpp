/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/evaluation/ClusteringAccuracy.h>

using namespace shogun;

float64_t CClusteringAccuracy::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	SGVector<int32_t> predicted_ilabels=predicted->get_int_labels();
	SGVector<int32_t> groundtruth_ilabels=ground_truth->get_int_labels();
	int32_t correct=0;
	for (int32_t i=0; i < predicted_ilabels.vlen; ++i)
	{
		if (predicted_ilabels[i] == groundtruth_ilabels[i])
			correct++;
	}
	return float64_t(correct)/predicted_ilabels.vlen;
}
