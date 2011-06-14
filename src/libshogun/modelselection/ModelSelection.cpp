/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/ModelSelection.h"
#include "modelselection/ModelSelectionParameters.h"
#include "evaluation/CrossValidation.h"

using namespace shogun;

CModelSelection::CModelSelection()
{
	m_model_parameters=NULL;
	m_cross_validation=NULL;
}

CModelSelection::CModelSelection(CModelSelectionParameters* model_parameters,
		CCrossValidation* cross_validation) :
	m_model_parameters(model_parameters), m_cross_validation(cross_validation)
{
	SG_REF(m_model_parameters);
	SG_REF(m_cross_validation);
}

CModelSelection::~CModelSelection()
{
	SG_UNREF(m_model_parameters);
	SG_UNREF(m_cross_validation);
}
