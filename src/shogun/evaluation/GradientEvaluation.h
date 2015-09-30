/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CGRADIENTEVALUATION_H_
#define CGRADIENTEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/evaluation/EvaluationResult.h>

namespace shogun
{

/** @brief Class evaluates a machine using its associated differentiable
 * function for the function value and its gradient with respect to parameters.
 */
class CGradientEvaluation: public CMachineEvaluation
{
public:
	/** default constructor */
	CGradientEvaluation();

	/** constructor
	 *
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	CGradientEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
			CEvaluation* evaluation_criterion, bool autolock=true);

	virtual ~CGradientEvaluation();

	/** returns the name of the machine evaluation
	 *
	 *  @return name GradientEvaluation
	 */
	virtual const char* get_name() const { return "GradientEvaluation"; }

	/** evaluates differentiable function for value and derivative.
	 *
	 * @return GradientResult containing value and gradient
	 */
	virtual CEvaluationResult* evaluate();

	/** set differentiable function
	*
	* @param diff differentiable function
	*/
	inline void set_function(CDifferentiableFunction* diff)
	{
		SG_REF(diff);
		SG_UNREF(m_diff);
		m_diff=diff;
	}

	/** get differentiable function
	*
	* @return differentiable function
	*/
	inline CDifferentiableFunction* get_function()
	{
		SG_REF(m_diff);
		return m_diff;
	}

private:
	/** initialses and registers parameters */
	void init();

	/** updates parameter dictionary of differentiable function */
	void update_parameter_dictionary();

private:
	/** differentiable function */
	CDifferentiableFunction* m_diff;

	/** parameter dictionary of differentiable function */
	CMap<TParameter*, CSGObject*>*  m_parameter_dictionary;
};
}
#endif /* CGRADIENTEVALUATION_H_ */
