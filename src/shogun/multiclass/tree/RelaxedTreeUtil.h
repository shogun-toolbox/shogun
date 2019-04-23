/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Bjoern Esser
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
	SGMatrix<float64_t> estimate_confusion_matrix(std::shared_ptr<BaseMulticlassMachine >machine, std::shared_ptr<Features >X, std::shared_ptr<MulticlassLabels >Y, int32_t num_classes);

	/**
	 * Get confusion matrix.
	 * @param conf_mat num_class-by-num_class matrix, confusion matrix will be assigned to this
	 * @param gt ground-truth labels
	 * @param pred predicted labels
	 */
	void get_confusion_matrix(SGMatrix<float64_t> &conf_mat, std::shared_ptr<MulticlassLabels >gt, std::shared_ptr<MulticlassLabels >pred);
};

} /* shogun */

#endif /* end of include guard: RELAXEDTREEUTIL_H__ */

