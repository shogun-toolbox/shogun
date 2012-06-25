/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */


#ifndef CGRADIENTEVALUATION_H_
#define CGRADIENTEVALUATION_H_

#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/evaluation/GradientResult.h>


namespace shogun
{
/** @brief GradientEvaluation evaluates a machine using
 * its associated differentiable function for the function
 * value and its gradient with respect to parameters.
 */

class CGradientEvaluation: public CMachineEvaluation
{

public:

	/*Constructor*/
	CGradientEvaluation();

	/** constructor
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	CGradientEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterion, bool autolock=true);

	/*Destructor*/
	virtual ~CGradientEvaluation();

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const
	{
		return "GradientEvaluation";
	}

	/*Evaluates differentiable function for value
	 * and derivative.
	 *
	 * @return GradientResult containing value and
	 * gradient
	 */
	virtual CEvaluationResult* evaluate();

	/** set Differentiable Function
	*
	* @param d Differentiable Function
	*/
	inline void set_function(CDifferentiableFunction* d) {m_diff = d;};

	/** get Differentiable Function
	*
	* @return Differentiable Function
	*/
	inline CDifferentiableFunction* get_function()
	{
		SG_REF(m_diff);
		return m_diff;
	};

private:

	CDifferentiableFunction* m_diff;
};

} /* namespace shogun */

#endif /* CGRADIENTEVALUATION_H_ */
