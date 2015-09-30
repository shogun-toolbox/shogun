/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash panda
 * Written(W) 2014 Abinash Panda
 */

#ifndef _MULTILABEL_ACCURACY__H__
#define _MULTILABEL_ACCURACY__H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

/** @brief Class CMultilabelAccuracy used to compute accuracy of multilabel
 * classification.
 *
 * Formally, for multilabel classification, the accuracy is defined as
 *
 * \f[
 *      $\frac{1}{p}\sum_{i=1}^{p}\frac{ |Y_i \cap h(x_i)|}{|Y_i \cup
 *      h(x_i)|}$
 * \f]
 */
class CMultilabelAccuracy : public CEvaluation
{
public:
	/** default constructor */
	CMultilabelAccuracy();

	/** destructor */
	virtual ~CMultilabelAccuracy();

	/** evaluate accuracy
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return accuracy
	 */
	virtual float64_t evaluate(CLabels * predicted, CLabels * ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

	/** @return name of the SGSerializable */
	virtual const char * get_name() const
	{
		return "MultilabelAccuracy";
	}
}; /* class CMultilabelAccuracy */

} /* namespace shogun */

#endif /* _MULTILABEL_ACCURACY__H__ */



