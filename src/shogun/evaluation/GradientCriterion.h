/*
 * GradientCriterion.h
 *
 *  Created on: Jun 27, 2012
 *      Author: jacobw
 */

#ifndef CGRADIENTCRITERION_H_
#define CGRADIENTCRITERION_H_

#include <shogun/evaluation/Evaluation.h>

namespace shogun
{

class CGradientCriterion: public CEvaluation
{

/* @brief CGradientCriterion
 *
 * Simple class which specifies the direction
 * of gradient search. Does not provide any
 * label evaluation measure, however.
 */
public:

	CGradientCriterion();

	virtual ~CGradientCriterion();

	/** evaluate labels (Not really used in this class).
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth) { return 0; }


	/** @return whether criterium has to be maximized or minimized */
	virtual EEvaluationDirection get_evaluation_direction() { return m_direction; }

	/** Set the evaluation direction
	 * @param evaluation direction to be set.
	 */
	virtual void set_evaluation_direction(EEvaluationDirection dir)
	{
		m_direction = dir;
	}

	/** get name */
	virtual inline const char* get_name() const { return "GradientCriterion"; }

private:

	EEvaluationDirection m_direction;

};

} /* namespace shogun */
#endif /* CGRADIENTCRITERION_H_ */
