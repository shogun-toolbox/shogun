/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/multiclass/tree/RelaxedTreeUtil.h>

using namespace shogun;

// TODO: here we accept a kernel instead of X?
SGMatrix<float64_t> RelaxedTreeUtil::estimate_confusion_matrix(CMulticlassMachine *machine, CFeatures *X, CMulticlassLabels *Y, int32_t num_classes)
{
	const int32_t N_splits = 5;
	CCrossValidationSplitting *split = new CCrossValidationSplitting(Y, N_splits);
	split->build_subsets();

	SGMatrix<float64_t> conf_mat(num_classes, num_classes), tmp_mat(num_classes, num_classes);
	conf_mat.zero();

	for (int32_t i=0; i < N_splits; ++i)
	{
	}

	SG_UNREF(split);
}

void RelaxedTreeUtil::get_confusion_matrix(SGMatrix<float64_t> &conf_mat, CMulticlassLabels *gt, CMulticlassLabels *pred)
{
	conf_mat.zero();

	for (index_t i=0; i < gt->get_num_labels(); ++i)
	{
		conf_mat(gt->get_int_label(i), pred->get_int_label(i)) += 1;
	}

	for (index_t i=0; i < conf_mat.num_rows; ++i)
	{
		float64_t n=0;
		for (index_t j=0; j < conf_mat.num_cols; ++j)
			n += conf_mat(i, j);
		if (n != 0)
			for (index_t j=0; j < conf_mat.num_cols; ++j)
				conf_mat(i, j) /= n;
	}
}
