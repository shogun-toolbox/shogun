/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MODELSELECTION_H_
#define __MODELSELECTION_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

class CModelSelectionParameters;
class CCrossValidation;
class CParameterCombination;

/**
 * @brief Abstract base class for model selection.
 * Takes a parameter tree which specifies parameters for model selection,
 * and a cross-validation instance and searches for the best combination of
 * parameters in the abstract method select_model(), which has to be implemented
 * in concrete sub-classes.
 */
class CModelSelection: public CSGObject
{
public:
	/** constructor
	 * @param model_parameters parameter tree with model parameters to optimize
	 * @param cross_validation cross-validation instance to use for evaluation
	 * of a certain combination of parameters
	 */
	CModelSelection(CModelSelectionParameters* model_parameters,
			CCrossValidation* cross_validation);

	/** destructor */
	virtual ~CModelSelection();

	/**
	 * abstract method to select model
	 *
	 * @param print if true, the current combination is printed
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print=false)=0;

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const	{ return "ModelSelection"; }

protected:
	/** model parameters */
	CModelSelectionParameters* m_model_parameters;
	/** cross validation */
	CCrossValidation* m_cross_validation;
};

}

#endif /* __MODELSELECTION_H_ */
