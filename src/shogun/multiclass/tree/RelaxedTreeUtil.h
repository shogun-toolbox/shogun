/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef RELAXEDTREEUTIL_H__
#define RELAXEDTREEUTIL_H__

#include <shogun/lib/config.h>
#include <shogun/machine/BaseMulticlassMachine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** Utility class for CRelaxedTree */
class RelaxedTreeUtil
{
public:

	/** estimate confusion matrix with cross validation.
	 * @param machine multiclass machine used to compute confusion matrix
	 * @param X training data
	 * @param Y labels for training data
	 * @param num_classes number of classes
	 * @return num_classes-by-num_classes confusion matrix.
	 */
	SGMatrix<float64_t> estimate_confusion_matrix(CBaseMulticlassMachine *machine, CFeatures *X, CMulticlassLabels *Y, int32_t num_classes);

	/**
	 * Get confusion matrix.
	 * @param conf_mat num_class-by-num_class matrix, confusion matrix will be assigned to this
	 * @param gt ground-truth labels
	 * @param pred predicted labels
	 */
	void get_confusion_matrix(SGMatrix<float64_t> &conf_mat, CMulticlassLabels *gt, CMulticlassLabels *pred);
};

} /* shogun */

#endif /* end of include guard: RELAXEDTREEUTIL_H__ */

