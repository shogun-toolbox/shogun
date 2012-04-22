/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2011 Soeren Sonnenburg
 * Written (W) 2012 Fernando José Iglesias García and Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn, Fernando José Iglesias Garcia
 */

#include <shogun/machine/MulticlassMachine.h>
#include <shogun/base/Parameter.h>
#include <shogun/features/Labels.h>

using namespace shogun;

CMulticlassMachine::CMulticlassMachine()
: CMachine(), m_multiclass_strategy(new CMulticlassOneVsRestStrategy()),
	m_machine(NULL), m_machines(new CDynamicObjectArray()),
	m_rejection_strategy(NULL)
{
	register_parameters();
}

CMulticlassMachine::CMulticlassMachine(
		CMulticlassStrategy *strategy,
		CMachine* machine, CLabels* labs)
: CMachine(), m_multiclass_strategy(strategy),
	m_machines(new CDynamicObjectArray()), m_rejection_strategy(NULL)
{
	set_labels(labs);
	SG_REF(machine);
	m_machine = machine;
	register_parameters();
}

CMulticlassMachine::~CMulticlassMachine()
{
	SG_UNREF(m_multiclass_strategy);
	SG_UNREF(m_rejection_strategy);
	SG_UNREF(m_machine);
	SG_UNREF(m_machines);
}

void CMulticlassMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_multiclass_strategy,"m_multiclass_type", "Multiclass strategy", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_machine, "m_machine", "The base machine", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_rejection_strategy, "m_rejection_strategy", "Rejection strategy", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_machines, "machines", "Machines that jointly make up the multi-class machine.", MS_NOT_AVAILABLE);
}

CLabels* CMulticlassMachine::apply(CFeatures* features)
{
	init_machines_for_apply(features);
	return apply();
}

CLabels* CMulticlassMachine::apply()
{
	/** Ensure that m_machines have the features set */
	init_machines_for_apply(NULL);

	switch (m_multiclass_strategy->get_strategy_type())
	{
		case ONE_VS_REST_STRATEGY:
			return classify_one_vs_rest();
		case ONE_VS_ONE_STRATEGY:
			return classify_one_vs_one();
		default:
			SG_ERROR("Unknown multiclass strategy\n");
	}

	return NULL;
}

bool CMulticlassMachine::train_machine(CFeatures* data)
{
	if ( !data && !is_ready() )
		SG_ERROR("Please provide training data.\n");
	else
		init_machine_for_train(data);

	m_machines->clear_array();
	CLabels *train_labels = new CLabels(get_num_rhs_vectors());
	SG_REF(train_labels);
	m_machine->set_labels(train_labels);

	m_multiclass_strategy->train_start(m_labels, train_labels);
	while (m_multiclass_strategy->train_has_more())
	{
		CSubset *subset=m_multiclass_strategy->train_prepare_next();
		if (subset)
		{
			train_labels->set_subset(subset);
			set_machine_subset(subset);
		}

		m_machine->train();
		m_machines->push_back(get_machine_from_trained(m_machine));

		if (subset)
		{
			train_labels->remove_subset();
			remove_machine_subset();

			subset_feats.destroy_vector();
			subset_labels.destroy_vector();
		}
	}

	m_multiclass_strategy->train_stop();
	SG_UNREF(train_labels);

	return true;
}

CLabels* CMulticlassMachine::classify_one_vs_rest()
{
	int32_t num_classes = m_labels->get_num_classes();
	int32_t num_machines = get_num_machines();
	ASSERT(num_machines==num_classes);
	CLabels* result=NULL;

	if (is_ready())
	{
		int32_t num_vectors = get_num_rhs_vectors();

		result=new CLabels(num_vectors);
		SG_REF(result);

		ASSERT(num_vectors==result->get_num_labels());
		CLabels** outputs=SG_MALLOC(CLabels*, num_machines);

		for (int32_t i=0; i<num_machines; i++)
		{
			CMachine *machine = (CMachine*)m_machines->get_element(i);
			ASSERT(machine);
			outputs[i]=machine->apply();
			SG_UNREF(machine);
		}

		SGVector<float64_t> outputs_for_i(num_machines);
		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<num_machines; j++)
				outputs_for_i[j] = outputs[j]->get_label(i);
			result->set_label(i, maxvote_one_vs_rest(outputs_for_i));
		}

		outputs_for_i.destroy_vector();

		for (int32_t i=0; i<num_machines; i++)
			SG_UNREF(outputs[i]);

		SG_FREE(outputs);
	}

	return result;
}

int32_t CMulticlassMachine::maxvote_one_vs_rest(const SGVector<float64_t> &predicts)
{
	int32_t winner = 0;

	if (m_rejection_strategy && m_rejection_strategy->reject(predicts))
	{
		winner=CLabels::REJECTION_LABEL;
	}
	else
	{
		float64_t max_out = predicts[0];

		for (int32_t j=1; j<predicts.vlen; j++)
		{
			if (predicts[j]>max_out)
			{
				max_out = predicts[j];
				winner = j;
			}
		}
	}
	return winner;
}

CLabels* CMulticlassMachine::classify_one_vs_one()
{
	int32_t num_classes  = m_labels->get_num_classes();
	int32_t num_machines = get_num_machines();
	if ( num_machines != num_classes*(num_classes-1)/2 )
		SG_ERROR("Dimension mismatch in classify_one_vs_one between number \
			of machines = %d and number of classes = %d\n", num_machines,
			num_classes);
	CLabels* result = NULL;

	if ( is_ready() )
	{
		int32_t num_vectors = get_num_rhs_vectors();

		result = new CLabels(num_vectors);
		SG_REF(result);

		ASSERT(num_vectors==result->get_num_labels());
		CLabels** outputs=SG_MALLOC(CLabels*, num_machines);

		for (int32_t i=0; i<num_machines; i++)
		{
			CMachine *machine = (CMachine*)m_machines->get_element(i);
			ASSERT(machine);
			outputs[i]=machine->apply();
			SG_UNREF(machine);
		}

		SGVector<float64_t> output_for_v(num_machines);

		for (int32_t v=0; v<num_vectors; v++)
		{
			for (int32_t i=0; i < num_machines; ++i)
				output_for_v[i] = outputs[i]->get_label(v);
			result->set_label(v, maxvote_one_vs_one(output_for_v, num_classes));
		}

		output_for_v.destroy_vector();

		for (int32_t i=0; i<num_machines; i++)
			SG_UNREF(outputs[i]);
		SG_FREE(outputs);
	}

	return result;
}

int32_t CMulticlassMachine::maxvote_one_vs_one(const SGVector<float64_t> &predicts, int32_t num_classes)
{
	int32_t s=0;
	SGVector<int32_t> votes(num_classes);
	votes.zero();

	for (int32_t i=0; i<num_classes; i++)
	{
		for (int32_t j=i+1; j<num_classes; j++)
		{
			if (predicts[s++]>0)
				votes[i]++;
			else
				votes[j]++;
		}
	}

	int32_t winner=0;
	int32_t max_votes=votes[0];

	for (int32_t i=1; i<num_classes; i++)
	{
		if (votes[i]>max_votes)
		{
			max_votes=votes[i];
			winner=i;
		}
	}

	votes.destroy_vector();

	return winner;
}


float64_t CMulticlassMachine::apply(int32_t num)
{
	init_machines_for_apply(NULL);
	if (m_multiclass_strategy->get_strategy_type()==ONE_VS_REST_STRATEGY)
		return classify_example_one_vs_rest(num);
	else if (m_multiclass_strategy->get_strategy_type()==ONE_VS_ONE_STRATEGY)
		return classify_example_one_vs_one(num);
	else
		SG_ERROR("unknown multiclass strategy\n");
	return 0;
}

float64_t CMulticlassMachine::classify_example_one_vs_rest(int32_t num)
{
	ASSERT(m_machines->get_num_elements()>0);
	SGVector<float64_t> outputs(m_machines->get_num_elements());

	for (int32_t i=0; i < m_machines->get_num_elements(); ++i)
	{
		CMachine *machine = get_machine(i);
		outputs[i]=machine->apply(num);
		SG_UNREF(machine);
	}

	float64_t winner = maxvote_one_vs_rest(outputs);
	outputs.destroy_vector();

	return winner;
}

float64_t CMulticlassMachine::classify_example_one_vs_one(int32_t num)
{
	int32_t num_classes=m_labels->get_num_classes();
	ASSERT(m_machines->get_num_elements()>0);
	ASSERT(m_machines->get_num_elements()==num_classes*(num_classes-1)/2);

	SGVector<float64_t> outputs(m_machines->get_num_elements());
	for (int32_t i=0; i < m_machines->get_num_elements(); ++i)
	{
		CMachine *machine = get_machine(i);
		outputs[i] = machine->apply(num);
		SG_UNREF(machine);
	}

	float64_t winner = maxvote_one_vs_one(outputs, num_classes);
	outputs.destroy_vector();

	return winner;
}
