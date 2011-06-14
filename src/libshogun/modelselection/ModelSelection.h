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

#include "base/SGObject.h"

namespace shogun
{

class CModelSelectionParameters;
class CCrossValidation;
class CParameterCombination;

class CModelSelection: public CSGObject
{
public:
	CModelSelection();
	CModelSelection(CModelSelectionParameters* model_parameters,
			CCrossValidation* cross_validation);
	virtual ~CModelSelection();

	virtual CParameterCombination* select_model()=0;

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const	{ return "ModelSelection"; }

protected:
	CModelSelectionParameters* m_model_parameters;
	CCrossValidation* m_cross_validation;
};

}

#endif /* __MODELSELECTION_H_ */
