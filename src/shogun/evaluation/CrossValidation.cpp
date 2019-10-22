/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Giovanni De Toni,
 *          Sergey Lisitsyn, Saurabh Mahindre, Jacob Walker, Viktor Gal,
 *          Leon Kuchenbecker
 */

#include <shogun/base/progress.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationStorage.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/View.h>

using namespace shogun;

CrossValidation::CrossValidation() : Seedable<MachineEvaluation>()
{
	init();
}

CrossValidation::CrossValidation(
    std::shared_ptr<Machine> machine, std::shared_ptr<Features> features, std::shared_ptr<Labels> labels,
    std::shared_ptr<SplittingStrategy> splitting_strategy, std::shared_ptr<Evaluation> evaluation_criterion)
    : Seedable<MachineEvaluation>(
          machine, features, labels, splitting_strategy, evaluation_criterion)
{
	init();
}

CrossValidation::CrossValidation(
    std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labels, std::shared_ptr<SplittingStrategy> splitting_strategy,
    std::shared_ptr<Evaluation> evaluation_criterion)
    : Seedable<MachineEvaluation>(
          machine, labels, splitting_strategy, evaluation_criterion)
{
	init();
}

CrossValidation::~CrossValidation()
{
}

void CrossValidation::init()
{
	m_num_runs = 1;

	SG_ADD(&m_num_runs, "num_runs", "Number of repetitions");
}

std::shared_ptr<EvaluationResult> CrossValidation::evaluate_impl() const
{
	SGVector<float64_t> results(m_num_runs);

	/* perform all the x-val runs */
	SG_DEBUG("starting {} runs of cross-validation", m_num_runs);
	for (auto i : SG_PROGRESS(range(m_num_runs)))
	{
		results[i] = evaluate_one_run(i);
		io::info("Result of cross-validation run {}/{} is {}", i+1, m_num_runs, results[i]);
	}

	/* construct evaluation result */
	auto result = std::make_shared<CrossValidationResult>();
	result->set_mean(Statistics::mean(results));
	if (m_num_runs > 1)
		result->set_std_dev(Statistics::std_deviation(results));
	else
		result->set_std_dev(0);

	return result;
}

void CrossValidation::set_num_runs(int32_t num_runs)
{
	if (num_runs < 1)
		error("{} is an illegal number of repetitions", num_runs);

	m_num_runs = num_runs;
}

float64_t CrossValidation::evaluate_one_run(int64_t index) const
{
	SG_TRACE("entering {}::evaluate_one_run()", get_name());
	index_t num_subsets = m_splitting_strategy->get_num_subsets();

	SG_DEBUG("building index sets for {}-fold cross-validation", num_subsets)
	m_splitting_strategy->build_subsets();

	SGVector<float64_t> results(num_subsets);

	#pragma omp parallel for shared(results)
	for (auto i = 0; i<num_subsets; ++i)
	{
		// only need to clone hyperparameters and settings of machine
		// model parameters are inferred/learned during training
		auto machine = make_clone(m_machine,
				ParameterProperties::HYPER | ParameterProperties::SETTING);

		SGVector<index_t> idx_train =
			m_splitting_strategy->generate_subset_inverse(i);

		SGVector<index_t> idx_test =
			m_splitting_strategy->generate_subset_indices(i);

		auto features_train = view(m_features, idx_train);
		auto labels_train = view(m_labels, idx_train);
		auto features_test = view(m_features, idx_test);
		auto labels_test = view(m_labels, idx_test);

		auto evaluation_criterion = make_clone(m_evaluation_criterion);

		machine->set_labels(labels_train);
		machine->train(features_train);

		auto result_labels = machine->apply(features_test);

		results[i] = evaluation_criterion->evaluate(result_labels, labels_test);
		io::info("Result of cross-validation fold {}/{} is {}", i+1, num_subsets, results[i]);
	}

	/* build arithmetic mean of results */
	float64_t mean = Statistics::mean(results);

	SG_TRACE("leaving {}::evaluate_one_run()", get_name());
	return mean;
}
