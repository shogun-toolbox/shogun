/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Giovanni De Toni,
 *          Sergey Lisitsyn, Saurabh Mahindre, Jacob Walker, Viktor Gal,
 *          Leon Kuchenbecker
 */

#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationStorage.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/lib/List.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/util/converters.h>
#include <shogun/util/factory.h>

using namespace shogun;

CrossValidation::CrossValidation() : MachineEvaluation()
{
	init();
}

CrossValidation::CrossValidation(
    std::shared_ptr<Machine> machine, std::shared_ptr<Features> features, std::shared_ptr<Labels> labels,
    std::shared_ptr<SplittingStrategy> splitting_strategy, std::shared_ptr<Evaluation> evaluation_criterion,
    bool autolock)
    : MachineEvaluation(
          machine, features, labels, splitting_strategy, evaluation_criterion,
          autolock)
{
	init();
}

CrossValidation::CrossValidation(
    std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labels, std::shared_ptr<SplittingStrategy> splitting_strategy,
    std::shared_ptr<Evaluation> evaluation_criterion, bool autolock)
    : MachineEvaluation(
          machine, labels, splitting_strategy, evaluation_criterion, autolock)
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

std::shared_ptr<EvaluationResult> CrossValidation::evaluate_impl()
{
	/* if for some reason the do_unlock_frag is set, unlock */
	if (m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock = false;
	}

	/* set labels in any case (no locking needs this) */
	m_machine->set_labels(m_labels);

	if (m_autolock)
	{
		/* if machine supports locking try to do so */
		if (m_machine->supports_locking())
		{
			/* only lock if machine is not yet locked */
			if (!m_machine->is_data_locked())
			{
				m_machine->data_lock(m_labels, m_features);
				m_do_unlock = true;
			}
		}
		else
		{
			SG_WARNING(
			    "%s does not support locking. Autolocking is skipped. "
			    "Set autolock flag to false to get rid of warning.\n",
			    m_machine->get_name());
		}
	}

	SGVector<float64_t> results(m_num_runs);

	/* perform all the x-val runs */
	SG_DEBUG("starting %d runs of cross-validation\n", m_num_runs)
	for (auto i : SG_PROGRESS(range(m_num_runs)))
	{
		/* evtl. update xvalidation output class */
		SG_DEBUG("Creating CrossValidationStorage.\n")
		auto storage = std::make_shared<CrossValidationStorage>();
		storage->put("num_runs", utils::safe_convert<index_t>(m_num_runs));
		storage->put(
		    "num_folds", utils::safe_convert<index_t>(
		                     m_splitting_strategy->get_num_subsets()));
		storage->put("labels", m_labels);
		storage->post_init();
		SG_DEBUG("Ending CrossValidationStorage initilization.\n")

		SG_DEBUG("entering cross-validation run %d \n", i)
		results[i] = evaluate_one_run(i, storage);
		SG_DEBUG("result of cross-validation run %d is %f\n", i, results[i])

		/* Emit the value */
		observe(
		    i, "cross_validation_run", "One run of CrossValidation",
		    storage->as<EvaluationResult>());
	}

	/* construct evaluation result */
	auto result = std::make_shared<CrossValidationResult>();
	result->set_mean(Statistics::mean(results));
	if (m_num_runs > 1)
		result->set_std_dev(Statistics::std_deviation(results));
	else
		result->set_std_dev(0);

	/* unlock machine if it was locked in this method */
	if (m_machine->is_data_locked() && m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock = false;
	}


	return result;
}

void CrossValidation::set_num_runs(int32_t num_runs)
{
	if (num_runs < 1)
		SG_ERROR("%d is an illegal number of repetitions\n", num_runs)

	m_num_runs = num_runs;
}

float64_t CrossValidation::evaluate_one_run(
    int64_t index, std::shared_ptr<CrossValidationStorage> storage)
{
	SG_DEBUG("entering %s::evaluate_one_run()\n", get_name())
	index_t num_subsets = m_splitting_strategy->get_num_subsets();

	SG_DEBUG("building index sets for %d-fold cross-validation\n", num_subsets)

	/* build index sets */
	m_splitting_strategy->build_subsets();

	/* results array */
	SGVector<float64_t> results(num_subsets);

	/* different behavior whether data is locked or not */
	if (m_machine->is_data_locked())
	{
		m_machine->set_store_model_features(true);
		SG_DEBUG("starting locked evaluation\n", get_name())
		/* do actual cross-validation */
		for (auto i : SG_PROGRESS(range(num_subsets)))
		{
			COMPUTATION_CONTROLLERS

			/* evtl. update xvalidation output class */
			auto fold = std::make_shared<CrossValidationFoldStorage>();
			fold->put("run_index", (index_t)index);
			fold->put("fold_index", i);

			/* index subset for training, will be freed below */
			SGVector<index_t> inverse_subset_indices =
			    m_splitting_strategy->generate_subset_inverse(i);

			/* train machine on training features */
			m_machine->train_locked(inverse_subset_indices);

			/* feature subset for testing */
			SGVector<index_t> subset_indices =
			    m_splitting_strategy->generate_subset_indices(i);

			/* evtl. update xvalidation output class */
			fold->put("train_indices", inverse_subset_indices);
			auto fold_machine = m_machine->clone()->as<Machine>();
			fold->put("trained_machine", fold_machine);

			/* produce output for desired indices */
			auto result_labels = m_machine->apply_locked(subset_indices);

			/* set subset for testing labels */
			m_labels->add_subset(subset_indices);

			/* evaluate against own labels */
			m_evaluation_criterion->set_indices(subset_indices);
			results[i] =
			    m_evaluation_criterion->evaluate(result_labels, m_labels);

			/* evtl. update xvalidation output class */
			fold->put("test_indices", subset_indices);
			fold->put("predicted_labels", result_labels);
			auto true_labels = m_labels->clone()->as<Labels>();
			fold->put("ground_truth_labels", true_labels);
			fold->post_update_results();
			fold->put("evaluation_result", results[i]);

			/* remove subset to prevent side effects */
			m_labels->remove_subset();

			/* Save fold into storage */
			storage->append_fold_result(fold);

			SG_DEBUG("done locked evaluation\n", get_name())
		}
	}
	else
	{
		SG_DEBUG("starting unlocked evaluation\n", get_name())
		/* tell machine to store model internally
		 * (otherwise changing subset of features will kaboom the classifier) */
		m_machine->set_store_model_features(true);

		/* do actual cross-validation */

		// TODO parallel xvalidation needs some serious fixing, see #3743
		//#pragma omp parallel for
		for (index_t i = 0; i < num_subsets; ++i)
		{
			COMPUTATION_CONTROLLERS

			auto fold = std::make_shared<CrossValidationFoldStorage>();


			auto machine = as_machine(m_machine->clone());

			// TODO while these are not used through const interfaces,
			// we unfortunately have to clone, even though these could be shared
			auto features = as_features(m_features->clone());
			auto labels = std::dynamic_pointer_cast<Labels>(m_labels->clone());
			auto evaluation_criterion =
			    as_evaluation(m_evaluation_criterion->clone());

			/* evtl. update xvalidation output class */
			fold->put("run_index", (index_t)index);
			fold->put("fold_index", i);

			/* set feature subset for training */
			SGVector<index_t> inverse_subset_indices =
			    m_splitting_strategy->generate_subset_inverse(i);

			features->add_subset(inverse_subset_indices);

			/* set label subset for training */
			labels->add_subset(inverse_subset_indices);

			SG_DEBUG("training set %d:\n", i)
			if (io->get_loglevel() == MSG_DEBUG)
			{
				SGVector<index_t>::display_vector(
				    inverse_subset_indices.vector, inverse_subset_indices.vlen,
				    "training indices");
			}

			/* train machine on training features and remove subset */
			SG_DEBUG("starting training\n")
			machine->set_labels(labels);
			machine->train(features);
			SG_DEBUG("finished training\n")

			/* evtl. update xvalidation output class */
			fold->put("train_indices", inverse_subset_indices);
			auto fold_machine = machine->clone()->as<Machine>();
			fold->put("trained_machine", fold_machine);

			features->remove_subset();
			labels->remove_subset();

			/* set feature subset for testing (subset method that stores
			 * pointer) */
			SGVector<index_t> subset_indices =
			    m_splitting_strategy->generate_subset_indices(i);
			features->add_subset(subset_indices);

			/* set label subset for testing */
			labels->add_subset(subset_indices);

			SG_DEBUG("test set %d:\n", i)
			if (io->get_loglevel() == MSG_DEBUG)
			{
				SGVector<index_t>::display_vector(
				    subset_indices.vector, subset_indices.vlen, "test indices");
			}

			/* apply machine to test features and remove subset */
			SG_DEBUG("starting evaluation\n")
			SG_DEBUG("%p\n", features.get())
			auto result_labels = machine->apply(features);
			SG_DEBUG("finished evaluation\n")
			features->remove_subset();


			/* evaluate */
			results[i] = evaluation_criterion->evaluate(result_labels, labels);
			SG_DEBUG("result on fold %d is %f\n", i, results[i])

			/* evtl. update xvalidation output class */
			fold->put("test_indices", subset_indices);
			fold->put("predicted_labels", result_labels);
			auto true_labels = labels->clone()->as<Labels>();
			fold->put("ground_truth_labels", true_labels);
			fold->post_update_results();
			fold->put("evaluation_result", results[i]);

			storage->append_fold_result(fold);

			/* clean up, remove subsets */
			labels->remove_subset();
		}

		SG_DEBUG("done unlocked evaluation\n", get_name())
	}

	/* build arithmetic mean of results */
	float64_t mean = Statistics::mean(results);

	SG_DEBUG("leaving %s::evaluate_one_run()\n", get_name())
	return mean;
}
