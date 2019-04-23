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
	PRCEvaluation() :
		BinaryClassEvaluation(), m_computed(false)
	{
		m_PRC_graph = SGMatrix<float64_t>();
		m_thresholds = SGVector<float64_t>();
		m_auPRC = 0.0;
	};

	/** destructor */
	virtual ~PRCEvaluation();

	/** get name */
	virtual const char* get_name() const { return "PRCEvaluation"; };

	/** evaluate PRC and auPRC
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return auPRC
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

	/** get auPRC
	 * @return area under PRC (auPRC)
	 */
	float64_t get_auPRC();

	/** get PRC
	 * precision is dim0 (x)
	 * recall is dim1 (y)
	 * @return PRC graph matrix
	 */
	SGMatrix<float64_t> get_PRC();

	/** get thresholds corresponding to points on the PRC graph
	 * @return thresholds
	 */
	SGVector<float64_t> get_thresholds();

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
