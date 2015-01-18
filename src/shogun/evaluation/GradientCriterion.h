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

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>

namespace shogun
{

/** @brief Simple class which specifies the direction of gradient search.
 *
 * Does not provide any label evaluation measure, however.
 */
class CGradientCriterion : public CEvaluation
{
public:
	/** default constructor */
	CGradientCriterion() : CEvaluation() { m_direction=ED_MINIMIZE; }

	virtual ~CGradientCriterion() { }

	/** evaluate labels (not really used in this class).
	 *
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth)
	{
		return 0.0;
	}

	/** @return whether criterion has to be maximized or minimized */
	virtual EEvaluationDirection get_evaluation_direction() const
	{
		return m_direction;
	}

	/** set the evaluation direction
	 *
	 * @param direction evaluation direction to be set
	 */
	virtual void set_evaluation_direction(EEvaluationDirection direction)
	{
		m_direction=direction;
	}

	/** returns name of evaluation criterion
	 *
	 * @return name GradientCriterion
	 */
	virtual const char* get_name() const { return "GradientCriterion"; }

private:
	/** evaluation direction */
	EEvaluationDirection m_direction;
};
}
#endif /* CGRADIENTCRITERION_H_ */
