/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#ifndef CGRADIENTMODELSELECTION_H_
#define CGRADIENTMODELSELECTION_H_

#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/modelselection/ModelSelection.h>
#include <shogun/base/DynArray.h>
#include <shogun/evaluation/GradientResult.h>


namespace shogun
{

/**
 * @brief Model selection class which searches for the best model by a gradient-
 * search.
 */
class CGradientModelSelection: public CModelSelection
{

public:

	/** constructor
	 * @param model_parameters
	 * @param cross_validation
	 */
	CGradientModelSelection(CModelSelectionParameters* model_parameters,
			CMachineEvaluation* machine_eval);

	/*Default Constructor*/
	CGradientModelSelection();

	/*Destructor*/
	virtual ~CGradientModelSelection();

	/**
	 * method to select model via gradient search
	 *
	 * @param print_state if true, the output is verbose
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print_state=false);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	inline virtual const char* get_name() {return "GradientModelSelection";}

	/* Set the maximum evaluations used in the optimization algorithm
	 *
	 * @param m max evaluations
	 */
	void set_max_evaluations(int m) {m_max_evaluations = m;}

	/* Get the maximum evaluations used in the optimization algorithm
	 *
	 * @return number of maximum evaluations
	 */
	int get_max_evaluations() {return m_max_evaluations;}

	/* Set the minimum level of gradient tolerance used in the
	 * optimization algorithm
	 *
	 * @param t tolerance level
	 */
	void set_grad_tolerance(float64_t t) {m_grad_tolerance = t;}

	/* Get the minimum level of gradient tolerance used in the
	 * optimization algorithm
	 *
	 * @return tolerance level
	 */
	float64_t get_grad_tolerance() {return m_grad_tolerance;}

private:

	/* nlopt callback function wrapper
	 *
	 * @param n number of parameters
	 *
	 * @param x vector of parameter values
	 *
	 * @param grad vector of gradient values with
	 * respect to parameter
	 *
	 * @param func_data data needed for the callback function. In this case,
	 * its a nlopt_package
	 *
	 * @return function value
	 */
	static double nlopt_function(unsigned n, const double *x, double *grad,
			void *func_data);

protected:

	/* struct used for nlopt callback function*/
	struct nlopt_package
	{
		shogun::CMachineEvaluation* m_machine_eval;
		shogun::CParameterCombination* m_current_combination;
		bool print_state;
	};

	/*Maximum number of evaluations used in optimization algorithm */
	int m_max_evaluations;

	/*Gradient tolerance used in optimization algorithm */
	float64_t m_grad_tolerance;

	/*Parameter combination tree*/
	CParameterCombination* m_current_combination;

};

}

#endif /* CGRADIENTMODELSELECTION_H_ */
