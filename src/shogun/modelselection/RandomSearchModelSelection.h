/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2011 Heiko Strathmann
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef RANDOMSEARCHMODELSELECTION_H_
#define RANDOMSEARCHMODELSELECTION_H_

#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ModelSelectionOutput.h>
#include <shogun/base/DynArray.h>

namespace shogun
{

class CModelSelectionParameters;

/**
 * @brief Model selection class which searches for the best model by a random
 * search. See CModelSelection for details.
 */
class CRandomSearchModelSelection: public CModelSelection
{
public:
	/** constructor */
	CRandomSearchModelSelection();

	/** constructor
	 * @param model_parameters
	 * @param cross_validation
	 * @param ratio
	 */
	CRandomSearchModelSelection(CModelSelectionParameters* model_parameters,
			CMachineEvaluation* machine_eval, float64_t ratio);

	/** destructor */
	virtual ~CRandomSearchModelSelection();

	/** get ratio */
	float64_t get_ratio() const { return m_ratio; };
	/** set ratio */
	void set_ratio(float64_t ratio) { REQUIRE(ratio>0.0 && ratio<1.0, "Ratio should be in ]0,1[ range"); m_ratio = ratio; };

	/**
	 * method to select model via grid search
	 *
	 * @param print_state if true, the current combination is printed
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print_state=false);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "RandomSearchModelSelection";
	}

protected:
	/** ratio */
	float64_t m_ratio;

};

}

#endif /* RANDOMSEARCHMODELSELECTION_H_ */
