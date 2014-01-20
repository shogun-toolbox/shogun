/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/lib/SGVector.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/ClusteringMutualInformation.h>

using namespace shogun;

float64_t CClusteringMutualInformation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted && ground_truth)
	ASSERT(predicted->get_label_type() == LT_MULTICLASS)
	ASSERT(ground_truth->get_label_type() == LT_MULTICLASS)
	SGVector<float64_t> label_p=((CMulticlassLabels*) predicted)->get_unique_labels();
	SGVector<float64_t> label_g=((CMulticlassLabels*) ground_truth)->get_unique_labels();

	if (label_p.vlen != label_g.vlen)
		SG_ERROR("Number of classes are different\n")
	index_t n_class=label_p.vlen;
	float64_t n_label=predicted->get_num_labels();

	SGVector<int32_t> ilabels_p=((CMulticlassLabels*) predicted)->get_int_labels();
	SGVector<int32_t> ilabels_g=((CMulticlassLabels*) ground_truth)->get_int_labels();

	SGMatrix<float64_t> G(n_class, n_class);
	for (index_t i=0; i < n_class; ++i)
	{
		for (index_t j=0; j < n_class; ++j)
			G(i, j)=find_match_count(ilabels_g, label_g[i],
				ilabels_p, label_p[j])/n_label;
	}

	SGVector<float64_t> G_rowsum(n_class);
	G_rowsum.zero();
	SGVector<float64_t> G_colsum(n_class);
	G_colsum.zero();
	for (index_t i=0; i < n_class; ++i)
	{
		for (index_t j=0; j < n_class; ++j)
		{
			G_rowsum[i] += G(i, j);
			G_colsum[i] += G(j, i);
		}
	}

	float64_t mutual_info = 0;
	for (index_t i=0; i < n_class; ++i)
	{
		for (index_t j=0; j < n_class; ++j)
		{
			if (G(i, j) != 0)
				mutual_info += G(i, j) * log(G(i,j) /
					(G_rowsum[i]*G_colsum[j]))/log(2.);
		}
	}

	float64_t entropy_p = 0;
	float64_t entropy_g = 0;
	for (index_t i=0; i < n_class; ++i)
	{
		entropy_g += -G_rowsum[i] * log(G_rowsum[i])/log(2.);
		entropy_p += -G_colsum[i] * log(G_colsum[i])/log(2.);
	}

	return mutual_info / CMath::max(entropy_g, entropy_p);
}
