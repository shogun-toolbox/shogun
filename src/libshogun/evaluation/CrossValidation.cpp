/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "evaluation/CrossValidation.h"
#include "machine/Machine.h"
#include "evaluation/Evaluation.h"
#include "evaluation/SplittingStrategy.h"
#include "base/Parameter.h"

using namespace shogun;

CCrossValidation::CCrossValidation()
{
	init();
}

CCrossValidation::~CCrossValidation()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_splitting_strategy);
	SG_UNREF(m_evaluation_criterium);
}

CCrossValidation::CCrossValidation(CMachine* machine, CFeatures* features,
		CLabels* labels, CSplittingStrategy* splitting_strategy,
		CEvaluation* evaluation_criterium)
{
	init();

	m_machine=machine;
	m_features=features;
	m_labels=labels;
	m_splitting_strategy=splitting_strategy;
	m_evaluation_criterium=evaluation_criterium;

	SG_REF(m_machine);
	SG_REF(m_features);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterium);
}

void CCrossValidation::init()
{
	m_machine=NULL;
	m_features=NULL;
	m_labels=NULL;
	m_splitting_strategy=NULL;
	m_evaluation_criterium=NULL;
	m_num_runs=1;
	m_conf_int_p=0;

	m_parameters->add((CSGObject**) &m_machine, "machine",
			"Used learning machine");
	m_parameters->add((CSGObject**) &m_features, "features", "Used features");
	m_parameters->add((CSGObject**) &m_labels, "labels", "Used labels");
	m_parameters->add((CSGObject**) &m_splitting_strategy,
			"splitting_strategy", "Used splitting strategy");
	m_parameters->add((CSGObject**) &m_evaluation_criterium,
			"evaluation_criterium", "Used evaluation criterium");
	m_parameters->add(&m_num_runs, "num_runs", "Number of repetitions");
	m_parameters->add(&m_conf_int_p, "conf_int_p", "p-value of confidence "
			"interval");
}

Parameter* CCrossValidation::get_machine_parameters() const
{
	return m_machine->m_parameters;
}

CrossValidationResult CCrossValidation::evaluate()
{
	float64_t* results=new float64_t[m_num_runs];

	for (index_t i=0; i<m_num_runs; ++i)
		results[i]=evaluate_one_run();

	/* construct evaluation result */
	CrossValidationResult result;
	result.value=CMath::mean(results, m_num_runs);
	result.has_conf_int=m_conf_int_p!=0;
	result.conf_int_p=m_conf_int_p;

	if (result.has_conf_int)
	{
		/* TODO: calculate confidence interval, maybe put this into CMath? */
		SG_NOTIMPLEMENTED;
	}
	else
	{
		result.conf_int_low=0;
		result.conf_int_up=0;
	}

	delete[] results;

	return result;
}

void CCrossValidation::set_conf_int_p(float64_t conf_int_p)
{
	if (conf_int_p<0||conf_int_p>=1)
	{
		SG_ERROR("%f is an illegal p-value for confidence interval of "
				"cross-validation\n", conf_int_p);
	}

	m_conf_int_p=conf_int_p;
}

void CCrossValidation::set_num_runs(int32_t num_runs)
{
	if (num_runs<1)
		SG_ERROR("%d is an illegal number of repetitions\n", num_runs);

	m_num_runs=num_runs;
}

float64_t CCrossValidation::evaluate_one_run()
{
	index_t num_subsets=m_splitting_strategy->get_num_subsets();
	float64_t* results=new float64_t[num_subsets];

	/* set labels to machine */
	m_machine->set_labels(m_labels);

	/* do actual cross-validation */
	for (index_t i=0; i<num_subsets; ++i)
	{
		/* set feature subset for training */
		SGVector<index_t> inverse_subset_indices=
				m_splitting_strategy->generate_subset_inverse(i);
		m_features->set_subset(new CSubset(inverse_subset_indices));

		/* set label subset for training (copy data before) */
		SGVector<index_t> inverse_subset_indices_copy(
				inverse_subset_indices.vlen);
		memcpy(inverse_subset_indices_copy.vector,
				inverse_subset_indices.vector,
				inverse_subset_indices.vlen*sizeof(index_t));
		m_labels->set_subset(new CSubset(inverse_subset_indices_copy));

		/* train machine on training features */
		m_machine->train(m_features);

		/* set feature subset for testing (subset method that stores pointer) */
		SGVector<index_t> subset_indices=
				m_splitting_strategy->generate_subset_indices(i);
		m_features->set_subset(new CSubset(subset_indices));

		/* apply machine to test features */
		CLabels* result_labels=m_machine->apply(m_features);
		SG_REF(result_labels);

		/* set label subset for testing (copy data before) */
		SGVector<index_t> subset_indices_copy(
				new index_t[subset_indices.vlen], subset_indices.vlen);
		memcpy(subset_indices_copy.vector, subset_indices.vector,
				subset_indices.vlen*sizeof(index_t));
		m_labels->set_subset(new CSubset(subset_indices_copy));

		/* evaluate */
		results[i]=m_evaluation_criterium->evaluate(result_labels, m_labels);

		/* clean up, reset subsets */
		SG_UNREF(result_labels);
		m_features->remove_subset();
		m_labels->remove_subset();
	}

	/* build arithmetic mean of results */
	float64_t mean=CMath::mean(results, num_subsets);

	/* clean up */
	delete[] results;

	return mean;
}
