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

	/** get number of true positives
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return vector of number of true positives per example
	 */
	virtual SGVector<int32_t> get_true_pos(CLabels * predicted,
			CLabels * ground_truth);

	/** get number of false positives
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return vector of number of false positives per example
	 */
	virtual SGVector<int32_t> get_false_pos(CLabels * predicted,
			CLabels * ground_truth);

	/** get number of false negatives
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return vector of number of false negatives per example
	 */
	virtual SGVector<int32_t> get_false_neg(CLabels * predicted,
			CLabels * ground_truth);

	/** get f1-score
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return f1-score
	 */
	virtual float64_t get_f1_score(CLabels * predicted, CLabels * ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

	/** @return name of the SGSerializable */
	virtual const char * get_name() const
	{
		return "MultilabelAccuracy";
	}

private:
	/** function to compute multi-label evaluation metrics
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 * @param accuracy accuracy to be computed
	 * @param score f1-score to be computed
	 * @param tp true positives to be computed
	 * @param fp false positives to be computed
	 * @param fn false negatives to be computed
	 */
	void evaluate(
			CLabels * predicted,
			CLabels * ground_truth,
			float64_t & accuracy,
			float64_t & score,
			SGVector<int32_t> & tp,
			SGVector<int32_t> & fp,
			SGVector<int32_t> & fn);

}; /* class CMultilabelAccuracy */

} /* namespace shogun */

#endif /* _MULTILABEL_ACCURACY__H__ */



