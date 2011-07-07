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

#include "modelselection/ModelSelection.h"
#include "base/DynArray.h"

namespace shogun
{

class CModelSelectionParameters;

class CGridSearchModelSelection: public CModelSelection
{
public:
	CGridSearchModelSelection();
	CGridSearchModelSelection(CModelSelectionParameters* model_parameters,
				CCrossValidation* cross_validation);

	virtual ~CGridSearchModelSelection();

	virtual CParameterCombination* select_model(float64_t& best_result);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "GridSearchModelSelection";
	}
};

}

#endif /* __GRIDSEARCHMODELSELECTION_H_ */
