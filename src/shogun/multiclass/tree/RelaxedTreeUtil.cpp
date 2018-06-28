/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/multiclass/tree/RelaxedTreeUtil.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

SGMatrix<float64_t> RelaxedTreeUtil::estimate_confusion_matrix(CBaseMulticlassMachine *machine, CFeatures *X, CMulticlassLabels *Y, int32_t num_classes)
{
	const int32_t N_splits = 2; // 5
	CCrossValidationSplitting *split = new CCrossValidationSplitting(Y, N_splits);
	split->build_subsets();

	SGMatrix<float64_t> conf_mat(num_classes, num_classes), tmp_mat(num_classes, num_classes);
	conf_mat.zero();

	machine->set_store_model_features(true);

	for (int32_t i=0; i < N_splits; ++i)
	{
		// subset for training
		SGVector<index_t> inverse_subset_indices = split->generate_subset_inverse(i);
		machine->set_labels(Y->view(inverse_subset_indices));
		machine->train(X->view(inverse_subset_indices));

		// subset for predicting
		SGVector<index_t> subset_indices = split->generate_subset_indices(i);
		auto feats_subset = wrap(X->view(subset_indices));
		auto labels_subset =
		    wrap(Y->view(subset_indices)->as<CMulticlassLabels>());
		CMulticlassLabels* pred = machine->apply_multiclass(feats_subset);

		get_confusion_matrix(tmp_mat, labels_subset, pred);

		for (index_t j=0; j < tmp_mat.num_rows; ++j)
		{
			for (index_t k=0; k < tmp_mat.num_cols; ++k)
			{
				conf_mat(j, k) += tmp_mat(j, k);
			}
		}

		SG_UNREF(pred);
	}

	SG_UNREF(split);

	for (index_t j=0; j < tmp_mat.num_rows; ++j)
	{
		for (index_t k=0; k < tmp_mat.num_cols; ++k)
		{
			conf_mat(j, k) /= N_splits;
		}
	}

	return conf_mat;
}

void RelaxedTreeUtil::get_confusion_matrix(SGMatrix<float64_t> &conf_mat, CMulticlassLabels *gt, CMulticlassLabels *pred)
{
	SGMatrix<int32_t> conf_mat_int = CMulticlassAccuracy::get_confusion_matrix(pred, gt);

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
