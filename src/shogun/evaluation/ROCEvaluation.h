/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Evan Shelhamer, 
 *          Roman Votyakov, Leon Kuchenbecker
 */

#ifndef ROCEVALUATION_H_
#define ROCEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/BinaryClassEvaluation.h>

namespace shogun
{

class Labels;

/** @brief Class ROCEvalution used to evaluate ROC
 * (Receiver Operating Characteristic) and an area
 * under ROC curve (auROC).
 *
 * Implementation is based on the efficient ROC algorithm as described in
 *
 * Fawcett, Tom (2004) ROC Graphs:
 * Notes and Practical Considerations for Researchers; Machine Learning, 2004
 */
class ROCEvaluation: public BinaryClassEvaluation
{
public:
	/** constructor */
	ROCEvaluation();

	/** destructor */
	virtual ~ROCEvaluation();

	/** get name */
	virtual const char* get_name() const { return "ROCEvaluation"; };

	/** evaluate ROC and auROC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auROC
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

	virtual EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

	/** get auROC
	 * @return area under ROC (auROC)
	 */
	float64_t get_auROC() const;

	/** get ROC
	 * @return ROC graph matrix
	 */
	SGMatrix<float64_t> get_ROC() const;

	/** get thresholds corresponding to points on the ROC graph
	 * @return thresholds
	 */
	SGVector<float64_t> get_thresholds() const;

protected:

	/** evaluate ROC and auROC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auROC
	 */
	float64_t evaluate_roc(const std::shared_ptr<BinaryLabels>& predicted, const std::shared_ptr<BinaryLabels>& ground_truth);

protected:

	/** 2-d array used to store ROC graph */
	SGMatrix<float64_t> m_ROC_graph;

	/** vector with thresholds corresponding to points on the ROC graph */
	SGVector<float64_t> m_thresholds;

	/** area under ROC graph */
	float64_t m_auROC;

	/** indicator of ROC and auROC being computed already */
	bool m_computed;
};

}

#endif /* ROCEVALUATION_H_ */
