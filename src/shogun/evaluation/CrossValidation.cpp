/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/modelselection/ModelSelectionOutput.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/ParameterMap.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

CCrossValidation::CCrossValidation()
{
	init();
}

CCrossValidation::CCrossValidation(CMachine* machine, CFeatures* features,
		CLabels* labels, CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock) :
		CMachineEvaluation(machine, features, labels, splitting_strategy,
		evaluation_criterion, autolock)
{
	init();
}

CCrossValidation::CCrossValidation(CMachine* machine, CLabels* labels,
		CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock) :
		CMachineEvaluation(machine, labels, splitting_strategy, evaluation_criterion,
		autolock)
{
	init();
}

CCrossValidation::~CCrossValidation()
{

}

void CCrossValidation::init()
{
	m_num_runs=1;
	m_conf_int_alpha=0;

	SG_ADD(&m_num_runs, "num_runs", "Number of repetitions",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_conf_int_alpha, "conf_int_alpha", "alpha-value " \
			"of confidence interval", MS_NOT_AVAILABLE);
}


CEvaluationResult* CCrossValidation::evaluate(CModelSelectionOutput* ms_output)
{
	SG_DEBUG("entering %s::evaluate()\n", get_name());

	/* if for some reason the do_unlock_frag is set, unlock */
	if (m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock=false;
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
				m_do_unlock=true;
			}
		}
		else
		{
			SG_WARNING("%s does not support locking. Autolocking is skipped. "
					"Set autolock flag to false to get rid of warning.\n",
					m_machine->get_name());
		}
	}

	SGVector<float64_t> results(m_num_runs);

	/* perform all the x-val runs */
	SG_DEBUG("starting %d runs of cross-validation\n", m_num_runs);
	for (index_t i=0; i <m_num_runs; ++i)
	{
		SG_DEBUG("entering cross-validation run %d \n", i);
		results[i]=evaluate_one_run(ms_output);
		SG_DEBUG("result of cross-validation run %d is %f\n", i, results[i]);
	}

	/* construct evaluation result */
	CrossValidationResult* result = new CrossValidationResult();
	result->has_conf_int=m_conf_int_alpha != 0;
	result->conf_int_alpha=m_conf_int_alpha;

	if (result->has_conf_int)
	{
		result->conf_int_alpha=m_conf_int_alpha;
		result->mean=CStatistics::confidence_intervals_mean(results,
				result->conf_int_alpha, result->conf_int_low, result->conf_int_up);
	}
	else
	{
		result->mean=CStatistics::mean(results);
		result->conf_int_low=0;
		result->conf_int_up=0;
	}

	/* unlock machine if it was locked in this method */
	if (m_machine->is_data_locked() && m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock=false;
	}

	SG_DEBUG("leaving %s::evaluate()\n", get_name());

	SG_REF(result);
	return result;
}

void CCrossValidation::set_conf_int_alpha(float64_t conf_int_alpha)
{
	if (conf_int_alpha <0 || conf_int_alpha>= 1) {
		SG_ERROR("%f is an illegal alpha-value for confidence interval of "
		"cross-validation\n", conf_int_alpha);
	}

	if (m_num_runs==1)
	{
		SG_WARNING("Confidence interval for Cross-Validation only possible"
				" when number of runs is >1, ignoring.\n");
	}
	else
		m_conf_int_alpha=conf_int_alpha;
}

void CCrossValidation::set_num_runs(int32_t num_runs)
{
	if (num_runs <1)
		SG_ERROR("%d is an illegal number of repetitions\n", num_runs);

	m_num_runs=num_runs;
}

float64_t CCrossValidation::evaluate_one_run(CModelSelectionOutput* ms_output)
{
	SG_DEBUG("entering %s::evaluate_one_run()\n", get_name());
	index_t num_subsets=m_splitting_strategy->get_num_subsets();

	SG_DEBUG("building index sets for %d-fold cross-validation\n", num_subsets);

	/* build index sets */
	m_splitting_strategy->build_subsets();

	/* results array */
	SGVector<float64_t> results(num_subsets);

	/* different behavior whether data is locked or not */
	if (m_machine->is_data_locked())
	{
		SG_DEBUG("starting locked evaluation\n", get_name());
		/* do actual cross-validation */
		for (index_t i=0; i <num_subsets; ++i)
		{
			/* index subset for training, will be freed below */
			SGVector<index_t> inverse_subset_indices =
					m_splitting_strategy->generate_subset_inverse(i);

			/* train machine on training features */
			m_machine->train_locked(inverse_subset_indices);

			/* feature subset for testing */
			SGVector<index_t> subset_indices =
					m_splitting_strategy->generate_subset_indices(i);

			if (ms_output)
			{
				ms_output->output_train_indices(inverse_subset_indices);
				ms_output->output_trained_machine(m_machine);
			}

			/* produce output for desired indices */
			CLabels* result_labels=m_machine->apply_locked(subset_indices);
			SG_REF(result_labels);

			/* set subset for testing labels */
			m_labels->add_subset(subset_indices);

			/* evaluate against own labels */
			results[i]=m_evaluation_criterion->evaluate(result_labels, m_labels);

			if (ms_output)
			{
				ms_output->output_test_indices(subset_indices);
				ms_output->output_test_result(result_labels);
				ms_output->output_test_true_result(m_labels);
				ms_output->output_evaluate_result(results[i]);
			}

			/* remove subset to prevent side effects */
			m_labels->remove_subset();

			/* clean up */
			SG_UNREF(result_labels);

			SG_DEBUG("done locked evaluation\n", get_name());
		}
	}
	else
	{
		SG_DEBUG("starting unlocked evaluation\n", get_name());
		/* tell machine to store model internally
		 * (otherwise changing subset of features will kaboom the classifier) */
		m_machine->set_store_model_features(true);

		/* do actual cross-validation */
		for (index_t i=0; i <num_subsets; ++i)
		{
			/* set feature subset for training */
			SGVector<index_t> inverse_subset_indices=
					m_splitting_strategy->generate_subset_inverse(i);
			ms_output->output_train_indices(inverse_subset_indices);
			m_features->add_subset(inverse_subset_indices);

			/* set label subset for training */
			m_labels->add_subset(inverse_subset_indices);

			SG_DEBUG("training set %d:\n", i);
			if (io->get_loglevel()==MSG_DEBUG)
			{
				SGVector<index_t>::display_vector(inverse_subset_indices.vector,
						inverse_subset_indices.vlen, "training indices");
			}

			/* train machine on training features and remove subset */
			m_machine->train(m_features);
			if (ms_output)
			{
				ms_output->output_train_indices(inverse_subset_indices);
				ms_output->output_trained_machine(m_machine);
			}

			m_features->remove_subset();
			m_labels->remove_subset();

			/* set feature subset for testing (subset method that stores pointer) */
			SGVector<index_t> subset_indices =
					m_splitting_strategy->generate_subset_indices(i);
			m_features->add_subset(subset_indices);

			/* set label subset for testing */
			m_labels->add_subset(subset_indices);

			SG_DEBUG("test set %d:\n", i);
			if (io->get_loglevel()==MSG_DEBUG)
			{
				SGVector<index_t>::display_vector(subset_indices.vector,
						subset_indices.vlen, "test indices");
			}

			/* apply machine to test features and remove subset */
			CLabels* result_labels=m_machine->apply(m_features);
			m_features->remove_subset();
			SG_REF(result_labels);

			/* evaluate */
			results[i]=m_evaluation_criterion->evaluate(result_labels, m_labels);
			if (ms_output)
			{
				ms_output->output_test_indices(subset_indices);
				ms_output->output_test_result(result_labels);
				ms_output->output_test_true_result(m_labels);
				ms_output->output_evaluate_result(results[i]);
			}
			SG_DEBUG("result on fold %d is %f\n", i, results[i]);

			/* clean up, remove subsets */
			SG_UNREF(result_labels);
			m_labels->remove_subset();
		}

		SG_DEBUG("done unlocked evaluation\n", get_name());
	}

	/* build arithmetic mean of results */
	float64_t mean=CStatistics::mean(results);

	SG_DEBUG("leaving %s::evaluate_one_run()\n", get_name());
	return mean;
}
