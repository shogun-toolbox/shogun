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

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/machine/MulticlassMachine.h>
#include <shogun/base/Parameter.h>
#include <shogun/features/Labels.h>

using namespace shogun;

CMulticlassMachine::CMulticlassMachine()
: CMachine(), m_multiclass_strategy(new CMulticlassOneVsRestStrategy()),
	m_machine(NULL), m_machines(new CDynamicObjectArray())
{
	SG_REF(m_multiclass_strategy);
	register_parameters();
}

CMulticlassMachine::CMulticlassMachine(
		CMulticlassStrategy *strategy,
		CMachine* machine, CLabels* labs)
: CMachine(), m_multiclass_strategy(strategy),
	m_machines(new CDynamicObjectArray())
{
	SG_REF(strategy);
	set_labels(labs);
	SG_REF(machine);
	m_machine = machine;
	register_parameters();
}

CMulticlassMachine::~CMulticlassMachine()
{
	SG_UNREF(m_multiclass_strategy);
	SG_UNREF(m_machine);
	SG_UNREF(m_machines);
}

void CMulticlassMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_multiclass_strategy,"m_multiclass_type", "Multiclass strategy", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_machine, "m_machine", "The base machine", MS_NOT_AVAILABLE);
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

	if (is_ready())
	{
		int32_t num_classes=m_labels->get_num_classes();
		int32_t num_vectors=get_num_rhs_vectors();
		int32_t num_machines=m_machines->get_num_elements();
		if (num_machines <= 0)
			SG_ERROR("num_machines = %d, did you train your machine?", num_machines);

		CLabels *result=new CLabels(num_vectors);
		SG_REF(result);
		CLabels **outputs=SG_MALLOC(CLabels*, num_machines);

		for (int32_t i=0; i < num_machines; ++i)
		{
			CMachine *machine = (CMachine*)m_machines->get_element(i);
			ASSERT(machine);
			outputs[i]=machine->apply();
			SG_UNREF(machine);
		}

		SGVector<float64_t> output_for_i(num_machines);
		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<num_machines; j++)
				output_for_i[j] = outputs[j]->get_label(i);

			result->set_label(i, m_multiclass_strategy->decide_label(output_for_i, num_classes));
		}

		output_for_i.destroy_vector();
		for (int32_t i=0; i < num_machines; ++i)
			SG_UNREF(outputs[i]);

		SG_FREE(outputs);
		
		return result;
	}
	else
	{
		return NULL;
	}
}

bool CMulticlassMachine::train_machine(CFeatures* data)
{
	ASSERT(m_multiclass_strategy);

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
		SGVector<index_t> subset=m_multiclass_strategy->train_prepare_next();
		if (subset.vlen)
		{
			train_labels->add_subset(subset);
			add_machine_subset(subset);
		}

		m_machine->train();
		m_machines->push_back(get_machine_from_trained(m_machine));

		if (subset.vlen)
		{
			train_labels->remove_subset();
			remove_machine_subset();
			subset.destroy_vector();
		}
	}

	m_multiclass_strategy->train_stop();
	SG_UNREF(train_labels);

	return true;
}

float64_t CMulticlassMachine::apply(int32_t num)
{
	init_machines_for_apply(NULL);

	ASSERT(m_machines->get_num_elements()>0);
	SGVector<float64_t> outputs(m_machines->get_num_elements());

	for (int32_t i=0; i < m_machines->get_num_elements(); ++i)
	{
		CMachine *machine = get_machine(i);
		outputs[i]=machine->apply(num);
		SG_UNREF(machine);
	}

	float64_t result=m_multiclass_strategy->decide_label(outputs, m_labels->get_num_classes());
	outputs.destroy_vector();

	return result;
}

