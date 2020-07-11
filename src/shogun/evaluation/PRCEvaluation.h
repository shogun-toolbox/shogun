/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Roman Votyakov, Evan Shelhamer, 
 *          Yuyu Zhang
 */

#ifndef PRCEVALUATION_H_
#define PRCEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/BinaryClassEvaluation.h>

namespace shogun
{

class Labels;

/** @brief Class PRCEvaluation used to evaluate PRC
 * (Precision Recall Curve) and an area under PRC curve (auPRC).
 *
 */
class PRCEvaluation: public BinaryClassEvaluation
{
public:
	/** constructor */
	PRCEvaluation();

	/** destructor */
	~PRCEvaluation() override;

	/** get name */
	const char* get_name() const override { return "PRCEvaluation"; };

	/** evaluate PRC and auPRC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auPRC
	 */
	float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth) override;

	inline EEvaluationDirection get_evaluation_direction() const override
	{
		return ED_MAXIMIZE;
	}

	/** get auPRC
	 * @return area under PRC (auPRC)
	 */
	float64_t get_auPRC() const;

	/** get PRC
	 * precision is dim0 (x)
	 * recall is dim1 (y)
	 * @return PRC graph matrix
	 */
	SGMatrix<float64_t> get_PRC() const;

	/** get thresholds corresponding to points on the PRC graph
	 * @return thresholds
	 */
	SGVector<float64_t> get_thresholds() const;

protected:

	/** 2-d array used to store PRC graph */
	SGMatrix<float64_t> m_PRC_graph;

	/** vector with thresholds corresponding to points on the PRC graph */
	SGVector<float64_t> m_thresholds;

	/** area under PRC graph */
	float64_t m_auPRC;

	/** indicator of PRC and auPRC being computed already */
	bool m_computed;
};

}

#endif /* PRCEVALUATION_H_ */
