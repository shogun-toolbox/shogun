/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2011 Soeren Sonnenburg
 * Written (W) 2012 Fernando José Iglesias García and Sergey Lisitsyn
 * Written (W) 2013 Shell Hu and Heiko Strathmann
 * Copyright (C) 2012 Sergey Lisitsyn, Fernando José Iglesias Garcia
 */

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/MulticlassMachine.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/labels/MulticlassMultipleOutputLabels.h>

using namespace shogun;

CMulticlassMachine::CMulticlassMachine()
: CBaseMulticlassMachine(), m_multiclass_strategy(new CMulticlassOneVsRestStrategy()),
	m_machine(NULL)
{
	SG_REF(m_multiclass_strategy);
	register_parameters();
}

CMulticlassMachine::CMulticlassMachine(
		CMulticlassStrategy *strategy,
		CMachine* machine, CLabels* labs)
: CBaseMulticlassMachine(), m_multiclass_strategy(strategy)
{
	SG_REF(strategy);
	set_labels(labs);
	SG_REF(machine);
	m_machine = machine;
	register_parameters();

	if (labs)
		init_strategy();
}

CMulticlassMachine::~CMulticlassMachine()
{
	SG_UNREF(m_multiclass_strategy);
	SG_UNREF(m_machine);
}

void CMulticlassMachine::set_labels(CLabels* lab)
{
    CMachine::set_labels(lab);
    if (lab)
        init_strategy();
}

void CMulticlassMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_multiclass_strategy,"m_multiclass_type", "Multiclass strategy", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_machine, "m_machine", "The base machine", MS_NOT_AVAILABLE);
}

void CMulticlassMachine::init_strategy()
{
    int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();
    m_multiclass_strategy->set_num_classes(num_classes);
}

CBinaryLabels* CMulticlassMachine::get_submachine_outputs(int32_t i)
{
	CMachine *machine = (CMachine*)m_machines->get_element(i);
	ASSERT(machine)
	CBinaryLabels* output = machine->apply_binary();
	SG_UNREF(machine);
	return output;
}

float64_t CMulticlassMachine::get_submachine_output(int32_t i, int32_t num)
{
	CMachine *machine = get_machine(i);
	float64_t output = 0.0;
	// dirty hack
	if (dynamic_cast<CLinearMachine*>(machine))
		output = ((CLinearMachine*)machine)->apply_one(num);
	if (dynamic_cast<CKernelMachine*>(machine))
		output = ((CKernelMachine*)machine)->apply_one(num);
	SG_UNREF(machine);
	return output;
}

CMulticlassLabels* CMulticlassMachine::apply_multiclass(CFeatures* data)
{
	SG_DEBUG("entering %s::apply_multiclass(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data);

	CMulticlassLabels* return_labels=NULL;

	if (data)
		init_machines_for_apply(data);
	else
		init_machines_for_apply(NULL);

	if (is_ready())
	{
		/* num vectors depends on whether data is provided */
		int32_t num_vectors=data ? data->get_num_vectors() :
				get_num_rhs_vectors();

		int32_t num_machines=m_machines->get_num_elements();
		if (num_machines <= 0)
			SG_ERROR("num_machines = %d, did you train your machine?", num_machines)

		CMulticlassLabels* result=new CMulticlassLabels(num_vectors);

		// if outputs are prob, only one confidence for each class
		int32_t num_classes=m_multiclass_strategy->get_num_classes();
		EProbHeuristicType heuris = get_prob_heuris();

		if (heuris!=PROB_HEURIS_NONE)
			result->allocate_confidences_for(num_classes);
		else
			result->allocate_confidences_for(num_machines);

		CBinaryLabels** outputs=SG_MALLOC(CBinaryLabels*, num_machines);
		SGVector<float64_t> As(num_machines);
		SGVector<float64_t> Bs(num_machines);

		for (int32_t i=0; i<num_machines; ++i)
		{
			outputs[i] = (CBinaryLabels*) get_submachine_outputs(i);

			if (heuris==OVA_SOFTMAX)
			{
				CStatistics::SigmoidParamters params = CStatistics::fit_sigmoid(outputs[i]->get_values());
				As[i] = params.a;
				Bs[i] = params.b;
			}

			if (heuris!=PROB_HEURIS_NONE && heuris!=OVA_SOFTMAX)
				outputs[i]->scores_to_probabilities(0,0);
		}

		SGVector<float64_t> output_for_i(num_machines);
		SGVector<float64_t> r_output_for_i(num_machines);
		if (heuris!=PROB_HEURIS_NONE)
			r_output_for_i.resize_vector(num_classes);

		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<num_machines; j++)
				output_for_i[j] = outputs[j]->get_value(i);

			if (heuris==PROB_HEURIS_NONE)
			{
				r_output_for_i = output_for_i;
			}
			else
			{
				if (heuris==OVA_SOFTMAX)
					m_multiclass_strategy->rescale_outputs(output_for_i,As,Bs);
				else
					m_multiclass_strategy->rescale_outputs(output_for_i);

				// only first num_classes are returned
				for (int32_t r=0; r<num_classes; r++)
					r_output_for_i[r] = output_for_i[r];

				SG_DEBUG("%s::apply_multiclass(): sum(r_output_for_i) = %f\n",
					get_name(), SGVector<float64_t>::sum(r_output_for_i.vector,num_classes));
			}

			// use rescaled outputs for label decision
			result->set_label(i, m_multiclass_strategy->decide_label(r_output_for_i));
			result->set_multiclass_confidences(i, r_output_for_i);
		}

		for (int32_t i=0; i < num_machines; ++i)
			SG_UNREF(outputs[i]);

		SG_FREE(outputs);

		return_labels=result;
	}
	else
		SG_ERROR("Not ready")


	SG_DEBUG("leaving %s::apply_multiclass(%s at %p)\n",
				get_name(), data ? data->get_name() : "NULL", data);
	return return_labels;
}

CMulticlassMultipleOutputLabels* CMulticlassMachine::apply_multiclass_multiple_output(CFeatures* data, int32_t n_outputs)
{
	CMulticlassMultipleOutputLabels* return_labels=NULL;

	if (data)
		init_machines_for_apply(data);
	else
		init_machines_for_apply(NULL);

	if (is_ready())
	{
		/* num vectors depends on whether data is provided */
		int32_t num_vectors=data ? data->get_num_vectors() :
				get_num_rhs_vectors();

		int32_t num_machines=m_machines->get_num_elements();
		if (num_machines <= 0)
			SG_ERROR("num_machines = %d, did you train your machine?", num_machines)
		REQUIRE(n_outputs<=num_machines,"You request more outputs than machines available")

		CMulticlassMultipleOutputLabels* result=new CMulticlassMultipleOutputLabels(num_vectors);
		CBinaryLabels** outputs=SG_MALLOC(CBinaryLabels*, num_machines);

		for (int32_t i=0; i < num_machines; ++i)
			outputs[i] = (CBinaryLabels*) get_submachine_outputs(i);

		SGVector<float64_t> output_for_i(num_machines);
		for (int32_t i=0; i<num_vectors; i++)
		{
			for (int32_t j=0; j<num_machines; j++)
				output_for_i[j] = outputs[j]->get_value(i);

			result->set_label(i, m_multiclass_strategy->decide_label_multiple_output(output_for_i, n_outputs));
		}

		for (int32_t i=0; i < num_machines; ++i)
			SG_UNREF(outputs[i]);

		SG_FREE(outputs);

		return_labels=result;
	}
	else
		SG_ERROR("Not ready")

	return return_labels;
}

bool CMulticlassMachine::train_machine(CFeatures* data)
{
	ASSERT(m_multiclass_strategy)

	if ( !data && !is_ready() )
		SG_ERROR("Please provide training data.\n")
	else
		init_machine_for_train(data);

	m_machines->reset_array();
	CBinaryLabels* train_labels = new CBinaryLabels(get_num_rhs_vectors());
	SG_REF(train_labels);
	m_machine->set_labels(train_labels);

	m_multiclass_strategy->train_start(CLabelsFactory::to_multiclass(m_labels), train_labels);
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
		}
	}

	m_multiclass_strategy->train_stop();
	SG_UNREF(train_labels);

	return true;
}

float64_t CMulticlassMachine::apply_one(int32_t vec_idx)
{
	init_machines_for_apply(NULL);

	ASSERT(m_machines->get_num_elements()>0)
	SGVector<float64_t> outputs(m_machines->get_num_elements());

	for (int32_t i=0; i<m_machines->get_num_elements(); i++)
		outputs[i] = get_submachine_output(i, vec_idx);

	float64_t result = m_multiclass_strategy->decide_label(outputs);

	return result;
}
