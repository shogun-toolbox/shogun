/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CGRADIENTMODELSELECTION_H_
#define CGRADIENTMODELSELECTION_H_

#include <lib/config.h>

#ifdef HAVE_NLOPT

#include <modelselection/ModelSelection.h>
#include <modelselection/ParameterCombination.h>

namespace shogun
{

/** @brief Model selection class which searches for the best model by a
 * gradient-search.
 */
class CGradientModelSelection : public CModelSelection
{
public:
	/** default constructor */
	CGradientModelSelection();

	/** constructor
	 *
	 * NOTE: if model_parameters is NULL, then gradient model selection is
	 * performed on all parameters of the machine.
	 *
	 * @param machine_eval machine evaluation object
	 * @param model_parameters parameters
	 */
	CGradientModelSelection(CMachineEvaluation* machine_eval,
			CModelSelectionParameters* model_parameters=NULL);

	virtual ~CGradientModelSelection();

	/** method to select model via gradient search
	 *
	 * @param print_state if true, the output is verbose
	 *
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print_state=false);

	/** returns the name of the model selection object
	 *
	 *  @return name GradientModelSelection
	 */
	virtual const char* get_name() const { return "GradientModelSelection"; }

	/** set the maximum number of evaluations used in the optimization algorithm
	 *
	 * @param max_evaluations maximum number of evaluations
	 */
	void set_max_evaluations(uint32_t max_evaluations)
	{
		m_max_evaluations=max_evaluations;
	}

	/** get the maximum number evaluations used in the optimization algorithm
	 *
	 * @return number of maximum evaluations
	 */
	uint32_t get_max_evaluations() const { return m_max_evaluations; }

	/** set the minimum level of gradient tolerance used in the optimization
	 * algorithm
	 *
	 * @param grad_tolerance tolerance level
	 */
	void set_grad_tolerance(float64_t grad_tolerance)
	{
		m_grad_tolerance=grad_tolerance;
	}

	/** get the minimum level of gradient tolerance used in the optimization
	 * algorithm
	 *
	 * @return tolerance level
	 */
	float64_t get_grad_tolerance() const { return m_grad_tolerance; }

private:
	/** initialize object */
	void init();

protected:
	/** maximum number of evaluations used in optimization algorithm */
	uint32_t m_max_evaluations;

	/** gradient tolerance used in optimization algorithm */
	float64_t m_grad_tolerance;
};
}
#endif /* HAVE_NLOPT */
#endif /* CGRADIENTMODELSELECTION_H_ */
