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

#include <shogun/modelselection/ModelSelection.h>
#include <shogun/base/DynArray.h>

namespace shogun
{

class CModelSelectionParameters;

/**
 * @brief Model selection class which searches for the best model by a grid-
 * search. See CModelSelection for details.
 */
class CGridSearchModelSelection: public CModelSelection
{
public:
	/** constructor */
	CGridSearchModelSelection();

	/** constructor
	 * @param model_parameters
	 * @param cross_validation
	 */
	CGridSearchModelSelection(CModelSelectionParameters* model_parameters,
				CCrossValidation* cross_validation);

	/** destructor */
	virtual ~CGridSearchModelSelection();

	/** select model */
	virtual CParameterCombination* select_model();

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "GridSearchModelSelection";
	}
};

}

#endif /* __GRIDSEARCHMODELSELECTION_H_ */
