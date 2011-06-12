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

using namespace shogun;

CCrossValidation::CCrossValidation()
{
	m_machine=NULL;
	m_features=NULL;
	m_labels=NULL;
	m_splitting_strategy=NULL;
	m_evaluation_criterium=NULL;
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
		CEvaluation* evaluation_criterium) :
	m_machine(machine), m_features(features), m_labels(labels),
			m_splitting_strategy(splitting_strategy),
			m_evaluation_criterium(evaluation_criterium)

{
	SG_REF(m_machine);
	SG_REF(m_features);
	SG_REF(m_labels);
	SG_REF(m_splitting_strategy);
	SG_REF(m_evaluation_criterium);
}

float64_t CCrossValidation::evaluate(int32_t num_runs, float64_t conf_int_p,
		float64_t* conf_int_low, float64_t* conf_int_up)
{
	if (num_runs<=0)
		SG_ERROR("number of cross-validation runs has to >0\n");

	/* check if confidence interval has to be computed */
	bool conf_interval=(conf_int_low!=NULL&&conf_int_up!=NULL);

	if (conf_interval&&(conf_int_p<=0||conf_int_p>=1))
	{
		SG_ERROR("illegal p-value for confidence interval of "
				"cross-validation\n");
	}

	float64_t* results=new float64_t[num_runs];

	for (index_t i=0; i<num_runs; ++i)
		results[i]=evaluate_one_run();

	/* TODO: calculate confidence interval, maybe put this into CMath? */
	if (conf_interval)
		SG_NOTIMPLEMENTED;

	float64_t mean=CMath::mean(results, num_runs);

	delete[] results;

	return mean;
}

float64_t CCrossValidation::evaluate_one_run()
{
	index_t num_subsets=m_splitting_strategy->get_num_subsets();
	float64_t* results=new float64_t[num_subsets];

	/* do actual cross-validation */
	for (index_t i=0; i<num_subsets; ++i)
	{
		/* set feature subset for training */
		SGVector<index_t> inverse_subset_indices;
		m_splitting_strategy->generate_subset_inverse(i, inverse_subset_indices);
		m_features->set_feature_subset(inverse_subset_indices.vector,
				inverse_subset_indices.length);

		/* train machine on training features */
		m_machine->train(m_features);

		/* set feature subset for testing */
		SGVector<index_t> subset_indices;
		m_splitting_strategy->generate_subset_indices(i, subset_indices);
		m_features->set_feature_subset(subset_indices.vector,
				subset_indices.length);

		/* apply machine to test features */
		CLabels* result_labels=m_machine->apply(m_features);

		/* evaluate */
		results[i]=m_evaluation_criterium->evaluate(result_labels, m_labels);
	}

	/* build arithmetic mean of results */
	float64_t mean=CMath::mean(results, num_subsets);
	delete[] results;

	return mean;
}
