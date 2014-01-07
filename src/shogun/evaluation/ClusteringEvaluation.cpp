/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <set>
#include <map>
#include <vector>
#include <algorithm>

#include <evaluation/ClusteringEvaluation.h>
#include <labels/MulticlassLabels.h>
#include <mathematics/munkres.h>

using namespace shogun;
using namespace std;

int32_t CClusteringEvaluation::find_match_count(SGVector<int32_t> l1, int32_t m1, SGVector<int32_t> l2, int32_t m2)
{
	int32_t match_count=0;
	for (int32_t i=l1.vlen-1; i >= 0; --i)
	{
		if (l1[i] == m1 && l2[i] == m2)
			match_count++;
	}

	return match_count;
}

int32_t CClusteringEvaluation::find_mismatch_count(SGVector<int32_t> l1, int32_t m1, SGVector<int32_t> l2, int32_t m2)
{
	return l1.vlen - find_match_count(l1, m1, l2, m2);
}

void CClusteringEvaluation::best_map(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels())
	ASSERT(predicted->get_label_type() == LT_MULTICLASS)
	ASSERT(ground_truth->get_label_type() == LT_MULTICLASS)

	SGVector<float64_t> label_p=((CMulticlassLabels*) predicted)->get_unique_labels();
	SGVector<float64_t> label_g=((CMulticlassLabels*) ground_truth)->get_unique_labels();

	SGVector<int32_t> predicted_ilabels=((CMulticlassLabels*) predicted)->get_int_labels();
	SGVector<int32_t> groundtruth_ilabels=((CMulticlassLabels*) ground_truth)->get_int_labels();

	int32_t n_class=max(label_p.vlen, label_g.vlen);
	SGMatrix<float64_t> G(n_class, n_class);
	G.zero();

	for (int32_t i=0; i < label_g.vlen; ++i)
	{
		for (int32_t j=0; j < label_p.vlen; ++j)
		{
			G(i, j)=find_mismatch_count(groundtruth_ilabels, static_cast<int32_t>(label_g[i]),
				predicted_ilabels, static_cast<int32_t>(label_p[j]));
		}
	}

	Munkres munkres_solver(G);
	munkres_solver.solve();

	std::map<int32_t, int32_t> label_map;
	for (int32_t i=0; i < label_p.vlen; ++i)
	{
		for (int32_t j=0; j < label_g.vlen; ++j)
		{
			if (G(j, i) == 0)
			{
				label_map.insert(make_pair(static_cast<int32_t>(label_p[i]),
						static_cast<int32_t>(label_g[j])));
				break;
			}
		}
	}

	for (int32_t i= 0; i < predicted_ilabels.vlen; ++i)
		((CMulticlassLabels*) predicted)->set_int_label(i, label_map[predicted_ilabels[i]]);
}
