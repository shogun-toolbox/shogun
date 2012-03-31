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

#include <shogun/evaluation/ClusteringEvaluation.h>
#include <shogun/mathematics/munkres.h>

using namespace shogun;
using namespace std;

static void unique_labels(CLabels* labels, vector<int32_t>& result)
{
	set<int32_t> uniq_lbl;
	for (int32_t i=labels->get_num_labels()-1; i >= 0; --i)
	{
		uniq_lbl.insert(labels->get_int_label(i));
	}
	result.assign(uniq_lbl.begin(), uniq_lbl.end());
}

static int32_t find_mismatch_count(const SGVector<int32_t>& l1, int32_t m1, const SGVector<int32_t>& l2, int32_t m2)
{
	int32_t match_count=0;
	for (int32_t i=l1.vlen-1; i >= 0; --i)
	{
		if (l1[i] != m1 || l2[i] != m2)
			match_count++;
	}

	return match_count;
}

void CClusteringEvaluation::best_map(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels());
	vector<int32_t> label_p, label_g;
	unique_labels(predicted, label_p);
	unique_labels(ground_truth, label_g);

	SGVector<int32_t> predicted_ilabels=predicted->get_int_labels();
	SGVector<int32_t> groundtruth_ilabels=ground_truth->get_int_labels();

	int32_t n_class=max(label_p.size(), label_g.size());
	SGMatrix<double> G(n_class, n_class);
	G.zero();

	for (size_t i=0; i < label_g.size(); ++i)
	{
		for (size_t j=0; j < label_p.size(); ++j)
		{
			G(i, j)=find_mismatch_count(groundtruth_ilabels, label_g[i],
				predicted_ilabels, label_p[j]);
		}
	}

	Munkres munkres_solver(G);
	munkres_solver.solve();

	map<int32_t, int32_t> label_map;
	for (size_t i=0; i < label_p.size(); ++i)
	{
		for (size_t j=0; j < label_g.size(); ++j)
		{
			if (G(j, i) == 0)
			{
				label_map.insert(make_pair(label_p[i], label_g[j]));
				break;
			}
		}
	}

	for (int32_t i= 0; i < predicted_ilabels.vlen; ++i)
		predicted_ilabels[i]=label_map[predicted_ilabels[i]];

	predicted->set_int_labels(predicted_ilabels);

	G.destroy_matrix();
}
