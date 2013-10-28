/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CModelSelection::CModelSelection()
{
	init();
}

CModelSelection::CModelSelection(CMachineEvaluation* machine_eval,
	CModelSelectionParameters* model_parameters)
{
	init();

	m_model_parameters=model_parameters;
	SG_REF(m_model_parameters);

	m_machine_eval=machine_eval;
	SG_REF(m_machine_eval);
}

void CModelSelection::init()
{
	m_model_parameters=NULL;
	m_machine_eval=NULL;

	SG_ADD((CSGObject**)&m_model_parameters, "model_parameters",
			"Parameter tree for model selection", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_machine_eval, "machine_evaluation",
			"Machine evaluation strategy", MS_NOT_AVAILABLE);
}

CModelSelection::~CModelSelection()
{
	SG_UNREF(m_model_parameters);
	SG_UNREF(m_machine_eval);
}
