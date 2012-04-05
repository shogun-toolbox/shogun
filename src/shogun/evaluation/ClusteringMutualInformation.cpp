/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <vector>
#include <cmath>
#include <algorithm>

#include <shogun/evaluation/ClusteringMutualInformation.h>

using namespace shogun;

float64_t CClusteringMutualInformation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	std::vector<int32_t> label_p=unique_labels(predicted);
	std::vector<int32_t> label_g=unique_labels(ground_truth);

	if (label_p.size() != label_g.size())
		SG_ERROR("Number of classes are different\n");
	uint32_t n_class=label_p.size();
	float64_t n_label=predicted->get_num_labels();

	SGVector<int32_t> ilabels_p=predicted->get_int_labels();
	SGVector<int32_t> ilabels_g=ground_truth->get_int_labels();

	SGMatrix<float64_t> G(n_class, n_class);
	for (size_t i=0; i < n_class; ++i)
	{
		for (size_t j=0; j < n_class; ++j)
			G(i, j)=find_match_count(ilabels_g, label_g[i],
				ilabels_p, label_p[j])/n_label;
	}

	std::vector<float64_t> G_rowsum(n_class), G_colsum(n_class);
	for (size_t i=0; i < n_class; ++i)
	{
		for (size_t j=0; j < n_class; ++j)
		{
			G_rowsum[i] += G(i, j);
			G_colsum[i] += G(j, i);
		}
	}

	float64_t mutual_info = 0;
	for (size_t i=0; i < n_class; ++i)
	{
		for (size_t j=0; j < n_class; ++j)
		{
			if (G(i, j) != 0)
				mutual_info += G(i, j) * log(G(i,j) /
					(G_rowsum[i]*G_colsum[j]))/log(2.);
		}
	}

	float64_t entropy_p = 0, entropy_g = 0;
	for (size_t i=0; i < n_class; ++i)
	{
		entropy_g += -G_rowsum[i] * log(G_rowsum[i])/log(2.);
		entropy_p += -G_colsum[i] * log(G_colsum[i])/log(2.);
	}

	return mutual_info / std::max(entropy_g, entropy_p);
}
