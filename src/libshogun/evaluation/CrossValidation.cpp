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

	m_parameters->add((CSGObject**) &m_machine, "machine",
			"Used learning machine");
	m_parameters->add((CSGObject**) &m_features, "features", "Used features");
	m_parameters->add((CSGObject**) &m_labels, "labels", "Used labels");
	m_parameters->add((CSGObject**) &m_splitting_strategy,
			"splitting_strategy", "Used splitting strategy");
	m_parameters->add((CSGObject**) &m_evaluation_criterium,
			"evaluation_criterium", "Used evaluation criterium");
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

	/* backup feature and label subsets */
	SGVector<index_t> feature_subset;
	SGVector<index_t> label_subset;
	feature_subset.vector=m_features->m_subset->get_subset(feature_subset.length);
	label_subset.vector=m_labels->m_subset->get_subset(label_subset.length);

	/* do actual cross-validation */
	for (index_t i=0; i<num_subsets; ++i)
	{
		/* set feature subset for training (use method that stores pointer) */
		SGVector<index_t> inverse_subset_indices;
		m_splitting_strategy->generate_subset_inverse(i, inverse_subset_indices);
		m_features->m_subset->set_subset(inverse_subset_indices.length,
				inverse_subset_indices.vector);

		/* train machine on training features */
		m_machine->train(m_features);

		/* set feature subset for testing (subset method that stores pointer) */
		SGVector<index_t> subset_indices;
		m_splitting_strategy->generate_subset_indices(i, subset_indices);
		m_features->m_subset->set_subset(subset_indices.length, subset_indices.vector);

		/* apply machine to test features */
		CLabels* result_labels=m_machine->apply(m_features);

		/* set label subset for testing (copy first index vector, subset method
		 * that stores pointer) */
		SGVector<index_t> subset_indices_copy(
				new index_t[subset_indices.length], subset_indices.length);
		memcpy(subset_indices_copy.vector, subset_indices.vector,
				subset_indices.length*sizeof(index_t));
		m_labels->m_subset->set_subset(subset_indices_copy.length,
				subset_indices_copy.vector);

		/* evaluate */
		results[i]=m_evaluation_criterium->evaluate(result_labels, m_labels);

		/* clean up, reset subsets */
		SG_UNREF(result_labels);
		m_features->m_subset->remove_subset();
		m_labels->m_subset->remove_subset();
	}

	/* restore feature and label subsets (use method that stores pointer) */
	m_features->m_subset->set_subset(feature_subset.length, feature_subset.vector);
	m_labels->m_subset->set_subset(label_subset.length, label_subset.vector);

	/* build arithmetic mean of results */
	float64_t mean=CMath::mean(results, num_subsets);

	/* clean up */
	delete[] results;

	return mean;
}
