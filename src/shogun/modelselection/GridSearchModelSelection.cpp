/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>

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

CParameterCombination* CGridSearchModelSelection::select_model(bool print)
{
	/* Retrieve all possible parameter combinations */
	CDynamicObjectArray<CParameterCombination>* combinations=
			m_model_parameters->get_combinations();

	CrossValidationResult best_result;

	CParameterCombination* best_combination=NULL;
	if (m_cross_validation->get_evaluation_direction()==ED_MAXIMIZE)
		best_result.mean=CMath::ALMOST_NEG_INFTY;
	else
		best_result.mean=CMath::ALMOST_INFTY;

	/* underlying learning machine */
	CMachine* machine=m_cross_validation->get_machine();

	/* apply all combinations and search for best one */
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* current_combination=combinations->get_element(i);

		/* eventually print */
		if (print)
		{
			SG_PRINT("trying combination:\n");
			current_combination->print_tree();
		}

		current_combination->apply_to_modsel_parameter(
				machine->m_model_selection_parameters);
		CrossValidationResult result=m_cross_validation->evaluate();

		/* check if current result is better, delete old combinations */
		if (m_cross_validation->get_evaluation_direction()==ED_MAXIMIZE)
		{
			if (result.mean>best_result.mean)
			{
				if (best_combination)
					SG_UNREF(best_combination);

				best_combination=combinations->get_element(i);
				best_result=result;
			}
			else
			{
				CParameterCombination* combination=combinations->get_element(i);
				SG_UNREF(combination);
			}
		}
		else
		{
			if (result.mean<best_result.mean)
			{
				if (best_combination)
					SG_UNREF(best_combination);

				best_combination=combinations->get_element(i);
				best_result=result;
			}
			else
			{
				CParameterCombination* combination=combinations->get_element(i);
				SG_UNREF(combination);
			}
		}

		SG_UNREF(current_combination);
	}

	SG_UNREF(machine);
	SG_UNREF(combinations);

	return best_combination;
}

