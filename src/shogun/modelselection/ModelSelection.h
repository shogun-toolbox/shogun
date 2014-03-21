/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTION_H_
#define __MODELSELECTION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/evaluation/MachineEvaluation.h>

namespace shogun
{
class CModelSelectionParameters;
class CParameterCombination;

/** @brief Abstract base class for model selection.
 *
 * Takes a parameter tree which specifies parameters for model selection, and a
 * cross-validation instance and searches for the best combination of parameters
 * in the abstract method select_model(), which has to be implemented in
 * concrete sub-classes.
 */
class CModelSelection: public CSGObject
{
public:
	/** default constructor */
	CModelSelection();

	/** constructor
	 *
	 * @param machine_eval object that computes the actual evaluation
	 * @param model_parameters parameter tree with model parameters to optimize
	 */
	CModelSelection(CMachineEvaluation* machine_eval,
			CModelSelectionParameters* model_parameters);

	/** destructor */
	virtual ~CModelSelection();

	/** abstract method to select model
	 *
	 * @param print_state if true, the current combination is printed
	 *
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print_state=false)=0;

private:
	/** initializer */
	void init();

protected:
	/** model parameters */
	CModelSelectionParameters* m_model_parameters;
	/** cross validation */
	CMachineEvaluation* m_machine_eval;
};
}
#endif /* __MODELSELECTION_H_ */
