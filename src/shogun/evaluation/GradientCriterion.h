/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CGRADIENTCRITERION_H_
#define CGRADIENTCRITERION_H_

#include <shogun/evaluation/Evaluation.h>

namespace shogun
{

class CGradientCriterion: public CEvaluation
{

/** @brief CGradientCriterion
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
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth)
	{ return 0; }


	/** @return whether criterium has to be maximized or minimized */
	virtual EEvaluationDirection get_evaluation_direction()
	{ return m_direction; }

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
