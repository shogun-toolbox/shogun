/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/CrossValidation.h>
#include <shogun/machine/Machine.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
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
		CEvaluation* evaluation_criterion, bool autolock)
{
	init();

	m_machine=machine;
	m_features=features;
	m_labels=labels;
	m_splitting_strategy=splitting_strategy;
	m_evaluation_criterion=evaluation_criterion;
	m_autolock=autolock;

	SG_REF(m_machine);
	SG_REF(m_features);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterion);
}

CCrossValidation::CCrossValidation(CMachine* machine, CLabels* labels,
		CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterion, bool autolock)
{
	init();

	m_machine=machine;
	m_labels=labels;
	m_splitting_strategy=splitting_strategy;
	m_evaluation_criterion=evaluation_criterion;
	m_autolock=autolock;

	SG_REF(m_machine);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterion);
}

CCrossValidation::~CCrossValidation()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_splitting_strategy);
	SG_UNREF(m_evaluation_criterion);
}

EEvaluationDirection CCrossValidation::get_evaluation_direction()
{
	return m_evaluation_criterion->get_evaluation_direction();
}

void CCrossValidation::init()
{
	m_machine=NULL;
	m_features=NULL;
	m_labels=NULL;
	m_splitting_strategy=NULL;
	m_evaluation_criterion=NULL;
	m_num_runs=1;
	m_conf_int_alpha=0;
	m_do_unlock=false;
	m_autolock=true;

	m_parameters->add((CSGObject**) &m_machine, "machine",
			"Used learning machine");
	m_parameters->add((CSGObject**) &m_features, "features", "Used features");
	m_parameters->add((CSGObject**) &m_labels, "labels", "Used labels");
	m_parameters->add((CSGObject**) &m_splitting_strategy, "splitting_strategy",
			"Used splitting strategy");
	m_parameters->add((CSGObject**) &m_evaluation_criterion,
			"evaluation_criterion", "Used evaluation criterion");
	m_parameters->add(&m_num_runs, "num_runs", "Number of repetitions");
	m_parameters->add(&m_conf_int_alpha, "conf_int_alpha",
			"alpha-value of confidence "
					"interval");
	m_parameters->add(&m_do_unlock, "do_unlock",
			"Whether machine should be unlocked after evaluation");
	m_parameters->add(&m_autolock, "m_autolock",
			"Whether machine should automatically try to be locked before "
			"evaluation");

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("m_do_unlock", CT_SCALAR, ST_NONE, PT_BOOL, 1),
			new SGParamInfo()
	);

	/* new parameter from param version 0 to 1 */
	m_parameter_map->put(
			new SGParamInfo("m_autolock", CT_SCALAR, ST_NONE, PT_BOOL, 1),
			new SGParamInfo()
	);
}

CMachine* CCrossValidation::get_machine() const
{
	SG_REF(m_machine);
	return m_machine;
}

CrossValidationResult CCrossValidation::evaluate()
{
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
	for (index_t i=0; i <m_num_runs; ++i)
		results.vector[i]=evaluate_one_run();

	/* construct evaluation result */
	CrossValidationResult result;
	result.has_conf_int=m_conf_int_alpha != 0;
	result.conf_int_alpha=m_conf_int_alpha;

	if (result.has_conf_int) {
		result.conf_int_alpha=m_conf_int_alpha;
		result.mean=CStatistics::confidence_intervals_mean(results,
				result.conf_int_alpha, result.conf_int_low, result.conf_int_up);
	}
	else
	{
		result.mean=CStatistics::mean(results);
		result.conf_int_low=0;
		result.conf_int_up=0;
	}

	SG_FREE(results.vector);

	/* unlock machine if it was locked in this method */
	if (m_machine->is_data_locked() && m_do_unlock)
	{
		m_machine->data_unlock();
		m_do_unlock=false;
	}

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

float64_t CCrossValidation::evaluate_one_run()
{
	index_t num_subsets=m_splitting_strategy->get_num_subsets();

	/* build index sets */
	m_splitting_strategy->build_subsets();

	/* results array */
	float64_t* results=SG_MALLOC(float64_t, num_subsets);

	/* different behavior whether data is locked or not */
	if (m_machine->is_data_locked())
	{
		/* do actual cross-validation */
		for (index_t i=0; i <num_subsets; ++i)
		{
			/* index subset for training, will be freed below */
			SGVector<index_t> inverse_subset_indices =
					m_splitting_strategy->generate_subset_inverse(i);

			/* train machine on training features */
			m_machine->train_locked(inverse_subset_indices);

			/* feature subset for testing, will be implicitly freed by CSubset */
			SGVector<index_t> subset_indices =
					m_splitting_strategy->generate_subset_indices(i);

			/* produce output for desired indices */
			CLabels* result_labels=m_machine->apply_locked(subset_indices);
			SG_REF(result_labels);

			/* set subset for training labels, note that this will (later) free
			 * the subset_indices vector */
			m_labels->set_subset(new CSubset(subset_indices));

			/* evaluate against own labels */
			results[i]=m_evaluation_criterion->evaluate(result_labels,
					m_labels);

			/* remove subset to prevent side efects */
			m_labels->remove_subset();

			/* clean up, inverse subset indices yet have to be deleted */
			SG_UNREF(result_labels);
			inverse_subset_indices.destroy_vector();
		}
	}
	else
	{
		/* tell machine to store model internally
		 * (otherwise changing subset of features will kaboom the classifier) */
		m_machine->set_store_model_features(true);

		/* do actual cross-validation */
		for (index_t i=0; i <num_subsets; ++i)
		{
			/* set feature subset for training */
			SGVector<index_t> inverse_subset_indices=
					m_splitting_strategy->generate_subset_inverse(i);
			CSubset* training_subset=new CSubset(inverse_subset_indices);
			m_features->add_subset(training_subset);

			/* set label subset for training */
			m_labels->set_subset(training_subset);

			/* train machine on training features and remove subset */
			m_machine->train(m_features);
			m_features->remove_subset();

			/* set feature subset for testing (subset method that stores pointer) */
			SGVector<index_t> subset_indices =
					m_splitting_strategy->generate_subset_indices(i);
			CSubset* test_subset=new CSubset(subset_indices);
			m_features->add_subset(test_subset);

			/* set label subset for testing */
			m_labels->set_subset(test_subset);

			/* apply machine to test features and remove subset */
			CLabels* result_labels=m_machine->apply(m_features);
			m_features->remove_subset();
			SG_REF(result_labels);

			/* evaluate */
			results[i]=m_evaluation_criterion->evaluate(result_labels, m_labels);

			/* clean up, remove subsets */
			SG_UNREF(result_labels);
			m_labels->remove_subset();
		}
	}

	/* build arithmetic mean of results */
	float64_t mean=CStatistics::mean(
			SGVector <float64_t> (results, num_subsets));

	/* clean up */
	SG_FREE(results);

	return mean;
}
