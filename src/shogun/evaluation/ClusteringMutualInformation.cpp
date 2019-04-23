/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang, Viktor Gal
 */

#include <shogun/lib/SGVector.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/ClusteringMutualInformation.h>

using namespace shogun;

float64_t ClusteringMutualInformation::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	ASSERT(predicted && ground_truth)
	ASSERT(predicted->get_label_type() == LT_MULTICLASS)
	ASSERT(ground_truth->get_label_type() == LT_MULTICLASS)
	SGVector<float64_t> label_p=multiclass_labels(predicted)->get_unique_labels();
	SGVector<float64_t> label_g=multiclass_labels(ground_truth)->get_unique_labels();

	if (label_p.vlen != label_g.vlen)
		SG_ERROR("Number of classes are different\n")
	index_t n_class=label_p.vlen;
	float64_t n_label=predicted->get_num_labels();

	SGVector<int32_t> ilabels_p=multiclass_labels(predicted)->get_int_labels();
	SGVector<int32_t> ilabels_g=multiclass_labels(ground_truth)->get_int_labels();

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

	return mutual_info / Math::max(entropy_g, entropy_p);
}
