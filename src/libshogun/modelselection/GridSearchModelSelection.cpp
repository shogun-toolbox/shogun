/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "modelselection/GridSearchModelSelection.h"
#include "modelselection/ParameterCombination.h"
#include "modelselection/ModelSelectionParameters.h"
#include "evaluation/CrossValidation.h"
#include "machine/Machine.h"

using namespace shogun;

CGridSearchModelSelection::CGridSearchModelSelection() :
	CModelSelection(NULL, NULL)
{

}

CGridSearchModelSelection::CGridSearchModelSelection(
		CModelSelectionParameters* model_parameters,
		CCrossValidation* cross_validation) :
	CModelSelection(model_parameters, cross_validation)
{

}

CGridSearchModelSelection::~CGridSearchModelSelection()
{
}

CParameterCombination* CGridSearchModelSelection::select_model(
		float64_t& best_result)
{
	/* Retrieve all possible parameter combinations */
	DynArray<CParameterCombination*>* combinations=
			m_model_parameters->get_combinations();

	CParameterCombination* best_combination=NULL;
	if (m_cross_validation->get_evaluation_direction()==ED_MAXIMIZE)
		best_result=CMath::ALMOST_NEG_INFTY;
	else
		best_result=CMath::ALMOST_INFTY;

	/* apply all combinations and search for best one */
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		combinations->get_element(i)->apply_to_parameter(m_cross_validation->get_machine_parameters());
		float64_t result=m_cross_validation->evaluate();

		/* check if current result is better, delete old combinations */
		if (m_cross_validation->get_evaluation_direction()==ED_MAXIMIZE)
		{
			if (result>best_result)
			{
				if (best_combination)
					best_combination->destroy(true, true);

				best_combination=combinations->get_element(i);
				best_result=result;
			}
			else
				combinations->get_element(i)->destroy(true, true);
		}
		else
		{
			if (result<best_result)
			{
				if (best_combination)
					best_combination->destroy(true, true);

				best_combination=combinations->get_element(i);
				best_result=result;
			}
			else
				combinations->get_element(i)->destroy(true, true);
		}
	}

	delete combinations;

	return best_combination;
}

