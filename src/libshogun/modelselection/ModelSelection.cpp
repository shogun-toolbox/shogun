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
#include "machine/Machine.h"
#include "evaluation/Evaluation.h"
#include "evaluation/SplittingStrategy.h"
#include "modelselection/ModelSelectionParameters.h"


using namespace shogun;

CModelSelection::CModelSelection()
{
	m_machine=NULL;
	m_model_parameters=NULL;
	m_splitting_strategy=NULL;
	m_evaluation_criterium=NULL;
	m_evaluation_direction=MS_UNDEFINED;
}

CModelSelection::CModelSelection(CMachine* machine,
		CEvaluation* evaluation_criterium, EEvaluationDirection eval_direction,
		CModelSelectionParameters* model_parameters,
		CSplittingStrategy* splitting_strategy) :
	m_machine(machine), m_evaluation_criterium(evaluation_criterium),
			m_evaluation_direction(eval_direction),
			m_model_parameters(model_parameters),
			m_splitting_strategy(splitting_strategy)
{
	SG_REF(m_machine);
	SG_REF(m_evaluation_criterium);
	SG_REF(m_model_parameters);
	SG_REF(m_splitting_strategy);
}

CModelSelection::~CModelSelection()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_evaluation_criterium);
	SG_UNREF(m_model_parameters);
	SG_UNREF(m_splitting_strategy);
}

CParameterCombination* CModelSelection::select_model()
{

}
