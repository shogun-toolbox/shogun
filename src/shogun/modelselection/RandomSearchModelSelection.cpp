/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann, Thoralf Klein,
 *          Soeren Sonnenburg, Sergey Lisitsyn, Roman Votyakov, Kyle McQuisten
 */

#ifdef ENABLE_MODEL_SELECTION
#include <shogun/base/progress.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/modelselection/RandomSearchModelSelection.h>

using namespace shogun;

RandomSearchModelSelection::RandomSearchModelSelection() : RandomMixin<ModelSelection>()
{
	set_ratio(0.5);
}

RandomSearchModelSelection::RandomSearchModelSelection(
		std::shared_ptr<MachineEvaluation> machine_eval,
		std::shared_ptr<ModelSelectionParameters> model_parameters, float64_t ratio)
		: RandomMixin<ModelSelection>(machine_eval, model_parameters)
{
	set_ratio(ratio);
}

RandomSearchModelSelection::~RandomSearchModelSelection()
{
}

std::shared_ptr<ParameterCombination> RandomSearchModelSelection::select_model(bool print_state)
{
	if (print_state)
		io::print("Generating parameter combinations\n");

	/* Retrieve all possible parameter combinations */
	auto all_combinations=
			m_model_parameters->get_combinations();

	int32_t n_all_combinations=all_combinations.size();
	SGVector<index_t> combinations_indices=Statistics::sample_indices(n_all_combinations*m_ratio, n_all_combinations, m_prng);

	std::vector<std::shared_ptr<ParameterCombination>> combinations;

	for (int32_t i=0; i<combinations_indices.vlen; i++)
		combinations.push_back(all_combinations[i]);

	auto best_result=std::make_shared<CrossValidationResult>();

	std::shared_ptr<ParameterCombination> best_combination=NULL;
	if (m_machine_eval->get_evaluation_direction()==ED_MAXIMIZE)
	{
		if (print_state) io::print("Direction is maximize\n");
		best_result->set_mean(Math::ALMOST_NEG_INFTY);
	}
	else
	{
		if (print_state) io::print("Direction is minimize\n");
		best_result->set_mean(Math::ALMOST_INFTY);
	}

	/* underlying learning machine */
	auto machine=m_machine_eval->get_machine();

	/* apply all combinations and search for best one */
	for (auto i : SG_PROGRESS(range(combinations.size())))
	{
		auto current_combination=
				combinations[i];

		/* eventually print */
		if (print_state)
		{
			io::print("trying combination:\n");
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
				best_combination=
						combinations[i];



				best_result=result;
			}
			else
			{
				auto combination=
						combinations[i];

			}
		}
		else
		{
			if (result->get_mean() < best_result->get_mean())
			{
				if (best_combination)


				best_combination=
						combinations[i];



				best_result=result;
			}
			else
			{
				auto combination=
						combinations[i];

			}
		}



	}

	return best_combination;
}
#endif
