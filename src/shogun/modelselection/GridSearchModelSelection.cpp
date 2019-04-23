/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Jacob Walker, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Giovanni De Toni, Thoralf Klein, Roman Votyakov, Kyle McQuisten
 */

#include <shogun/base/progress.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>

using namespace shogun;

GridSearchModelSelection::GridSearchModelSelection() : ModelSelection()
{
}

GridSearchModelSelection::GridSearchModelSelection(
		std::shared_ptr<MachineEvaluation> machine_eval,
		std::shared_ptr<ModelSelectionParameters> model_parameters)
		: ModelSelection(machine_eval, model_parameters)
{
}

GridSearchModelSelection::~GridSearchModelSelection()
{
}

std::shared_ptr<ParameterCombination> GridSearchModelSelection::select_model(bool print_state)
{
	if (print_state)
		SG_PRINT("Generating parameter combinations\n")

	/* Retrieve all possible parameter combinations */
	auto combinations=
			m_model_parameters->get_combinations()->as<DynamicObjectArray>();

	auto best_result=std::make_shared<CrossValidationResult>();

	std::shared_ptr<ParameterCombination> best_combination=NULL;
	if (m_machine_eval->get_evaluation_direction()==ED_MAXIMIZE)
	{
		if (print_state) SG_PRINT("Direction is maximize\n")
		best_result->set_mean(Math::ALMOST_NEG_INFTY);
	}
	else
	{
		if (print_state) SG_PRINT("Direction is minimize\n")
		best_result->set_mean(Math::ALMOST_INFTY);
	}

	/* underlying learning machine */
	auto machine=m_machine_eval->get_machine();

	/* apply all combinations and search for best one */
	for (auto i : SG_PROGRESS(range(combinations->get_num_elements())))
	{
		auto current_combination=combinations->get_element<ParameterCombination>(i);

		/* eventually print */
		if (print_state)
		{
			SG_PRINT("trying combination:\n")
			current_combination->print_tree();
		}

		current_combination->apply_to_modsel_parameter(
				machine->m_model_selection_parameters);

		/* note that this may implicitly lock and unlockthe machine */
		auto result =
		    m_machine_eval->evaluate()->as<CrossValidationResult>();

		if (print_state)
			result->print_result();

		/* check if current result is better, delete old combinations */
		if (m_machine_eval->get_evaluation_direction()==ED_MAXIMIZE)
		{
			if (result->get_mean() > best_result->get_mean())
			{
				if (best_combination)


				best_combination=
						combinations->get_element<ParameterCombination>(i);



				best_result=result;
			}
			else
			{
				auto combination=
						combinations->get_element<ParameterCombination>(i);

			}
		}
		else
		{
			if (result->get_mean() < best_result->get_mean())
			{
				if (best_combination)


				best_combination=
						combinations->get_element<ParameterCombination>(i);



				best_result=result;
			}
			else
			{
				auto combination=
						combinations->get_element<ParameterCombination>(i);

			}
		}



	}


	return best_combination;
}
