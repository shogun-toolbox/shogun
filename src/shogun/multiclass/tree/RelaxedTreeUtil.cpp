/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/lib/View.h>
#include <shogun/multiclass/tree/RelaxedTreeUtil.h>

using namespace shogun;

SGMatrix<float64_t> RelaxedTreeUtil::estimate_confusion_matrix(std::shared_ptr<BaseMulticlassMachine >machine, std::shared_ptr<Features >X, std::shared_ptr<MulticlassLabels >Y, int32_t num_classes)
{
	const int32_t N_splits = 2; // 5
	auto split = std::make_shared<CrossValidationSplitting>(Y, N_splits);
	split->build_subsets();

	SGMatrix<float64_t> conf_mat(num_classes, num_classes), tmp_mat(num_classes, num_classes);
	conf_mat.zero();

	for (int32_t i=0; i < N_splits; ++i)
	{
		// subset for training
		SGVector<index_t> inverse_subset_indices = split->generate_subset_inverse(i);
		machine->set_labels(view(Y, inverse_subset_indices));
		machine->train(view(X, inverse_subset_indices));

		// subset for predicting
		SGVector<index_t> subset_indices = split->generate_subset_indices(i);
		auto feats_subset = view(X, subset_indices);
		auto labels_subset = view(Y, subset_indices);
		auto pred = machine->apply_multiclass(feats_subset);

		get_confusion_matrix(tmp_mat, labels_subset, pred);

		for (index_t j=0; j < tmp_mat.num_rows; ++j)
		{
			for (index_t k=0; k < tmp_mat.num_cols; ++k)
			{
				conf_mat(j, k) += tmp_mat(j, k);
			}
		}


	}



	for (index_t j=0; j < tmp_mat.num_rows; ++j)
	{
		for (index_t k=0; k < tmp_mat.num_cols; ++k)
		{
			conf_mat(j, k) /= N_splits;
		}
	}

	return conf_mat;
}

void RelaxedTreeUtil::get_confusion_matrix(SGMatrix<float64_t> &conf_mat, std::shared_ptr<MulticlassLabels >gt, std::shared_ptr<MulticlassLabels >pred)
{
	SGMatrix<int32_t> conf_mat_int = MulticlassAccuracy::get_confusion_matrix(pred, gt);

	for (index_t i=0; i < conf_mat.num_rows; ++i)
	{
		float64_t n=0;
		for (index_t j=0; j < conf_mat.num_cols; ++j)
		{
			conf_mat(i, j) = conf_mat_int(i, j);
			n += conf_mat(i, j);
		}

		if (n != 0)
		{
			for (index_t j=0; j < conf_mat.num_cols; ++j)
				conf_mat(i, j) /= n;
		}
	}
}
