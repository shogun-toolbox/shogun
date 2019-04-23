/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Bjoern Esser
 */

#include <set>
#include <map>
#include <vector>
#include <algorithm>

#include <shogun/evaluation/ClusteringEvaluation.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/munkres.h>

using namespace shogun;
using namespace std;

ClusteringEvaluation::ClusteringEvaluation() : Evaluation()
{
	m_use_best_map = true;
	SG_ADD(&m_use_best_map, "use_best_map",
		"Find best match between predicted labels and the ground truth");
}

float64_t ClusteringEvaluation::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	if(m_use_best_map)
		predicted = best_map(predicted, ground_truth);
	float64_t result = evaluate_impl(predicted,ground_truth);
	return result;
}


int32_t ClusteringEvaluation::find_match_count(SGVector<int32_t> l1, int32_t m1, SGVector<int32_t> l2, int32_t m2)
{
	int32_t match_count=0;
	for (int32_t i=l1.vlen-1; i >= 0; --i)
	{
		if (l1[i] == m1 && l2[i] == m2)
			match_count++;
	}

	return match_count;
}

int32_t ClusteringEvaluation::find_mismatch_count(SGVector<int32_t> l1, int32_t m1, SGVector<int32_t> l2, int32_t m2)
{
	return l1.vlen - find_match_count(l1, m1, l2, m2);
}

std::shared_ptr<Labels> ClusteringEvaluation::best_map(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels())
	ASSERT(predicted->get_label_type() == LT_MULTICLASS)
	ASSERT(ground_truth->get_label_type() == LT_MULTICLASS)

	SGVector<float64_t> label_p=multiclass_labels(predicted)->get_unique_labels();
	SGVector<float64_t> label_g=multiclass_labels(ground_truth)->get_unique_labels();

	SGVector<int32_t> predicted_ilabels=multiclass_labels(predicted)->get_int_labels();
	SGVector<int32_t> groundtruth_ilabels=multiclass_labels(ground_truth)->get_int_labels();

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

	auto result = std::make_shared<MulticlassLabels>(predicted->get_num_labels());
	for (int32_t i= 0; i < predicted_ilabels.vlen; ++i)
		result->set_int_label(i, label_map[predicted_ilabels[i]]);

	return result;
}
