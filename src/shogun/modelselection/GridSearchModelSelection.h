/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __GRIDSEARCHMODELSELECTION_H_
#define __GRIDSEARCHMODELSELECTION_H_

#include <modelselection/ModelSelection.h>
#include <base/DynArray.h>

namespace shogun
{
class CModelSelectionParameters;

/** @brief Model selection class which searches for the best model by a grid-
 * search. See CModelSelection for details.
 */
class CGridSearchModelSelection : public CModelSelection
{
public:
	/** constructor */
	CGridSearchModelSelection();

	/** constructor
	 *
	 * @param machine_eval machine evaluation object
	 * @param model_parameters parameters
	 */
	CGridSearchModelSelection(CMachineEvaluation* machine_eval,
			CModelSelectionParameters* model_parameters);

	/** destructor */
	virtual ~CGridSearchModelSelection();

	/** method to select model via grid search
	 *
	 * @param print_state if true, the current combination is printed
	 *
	 * @return best combination of model parameters
	 */
	virtual CParameterCombination* select_model(bool print_state=false);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const { return "GridSearchModelSelection"; }
};
}
#endif /* __GRIDSEARCHMODELSELECTION_H_ */
