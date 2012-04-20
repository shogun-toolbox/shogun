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
: CMachine(), m_multiclass_strategy(ONE_VS_REST_STRATEGY),
  m_machine(NULL), m_rejection_strategy(NULL)
{
	register_parameters();
}

CMulticlassMachine::CMulticlassMachine(
	EMulticlassStrategy strategy,
	CMachine* machine, CLabels* labs)
: CMachine(), m_multiclass_strategy(strategy), m_rejection_strategy(NULL)
{
	set_labels(labs);
	SG_REF(machine);
	m_machine = machine;
	register_parameters();
}

CMulticlassMachine::~CMulticlassMachine()
{
	SG_UNREF(m_rejection_strategy);
	SG_UNREF(m_machine);

	clear_machines();
}

void CMulticlassMachine::register_parameters()
{
	SG_ADD((machine_int_t*)&m_multiclass_strategy,"m_multiclass_type", "Multiclass strategy", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_machine, "m_machine", "The base machine", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_rejection_strategy, "m_rejection_strategy", "Rejection strategy", MS_NOT_AVAILABLE);
	//TODO: fix this
	//SG_ADD(&m_machines, "machines", "Machines that jointly make up the multi-class machine.", MS_NOT_AVAILABLE);
}

void CMulticlassMachine::clear_machines()
{
	m_machines.clear_array();
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

	switch (m_multiclass_strategy)
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

	switch (m_multiclass_strategy)
	{
		case ONE_VS_REST_STRATEGY:
			return train_one_vs_rest();
		case ONE_VS_ONE_STRATEGY:
			return train_one_vs_one();
		default:
			SG_ERROR("Unknown multiclass strategy\n");
	}

	return false;
}

bool CMulticlassMachine::train_one_vs_rest()
{
	int32_t num_classes = m_labels->get_num_classes();
	int32_t num_vectors = get_num_rhs_vectors();

	clear_machines();
	m_machines.clear_array();
	CLabels* train_labels = new CLabels(num_vectors);
	SG_REF(train_labels);
	m_machine->set_labels(train_labels);

	for (int32_t i=0; i<num_classes; i++)
	{
		for (int32_t j=0; j<num_vectors; j++)
		{
			if (m_labels->get_int_label(j)==i)
				train_labels->set_label(j,+1.0);
			else
				train_labels->set_label(j,-1.0);
		}

		m_machine->train();
		
		m_machines.push_back(get_machine_from_trained(m_machine));
	}

	SG_UNREF(train_labels);
	return true;
}

bool CMulticlassMachine::train_one_vs_one()
{
	int32_t num_classes = m_labels->get_num_classes();
	int32_t num_vectors = get_num_rhs_vectors();

	clear_machines();
	m_machines.clear_array();
	CLabels* train_labels = new CLabels(num_vectors);
	SG_REF(train_labels);
	m_machine->set_labels(train_labels);

	/** Number of vectors included in every subset */
	int32_t tot = 0;

	/** Train each machine */
	for (int32_t i=0; i<num_classes; i++)
	{
		for (int32_t j=i+1; j<num_classes; j++)
		{
			SGVector<index_t> subset_labels(num_vectors);
			SGVector<index_t> subset_feats(num_vectors);

			/** Modify the labels of the training examples that belong
			 *  to the classes relevant to train with this machine.
			 *  The training examples of the other classes are excluded */

			tot = 0;
			for (int32_t k=0; k<num_vectors; k++)
			{
				/* It is important to use the same index-label association
				 * here and in classifiy_one_vs_one: i -> 1.0, j -> -1.0 */

				if (m_labels->get_int_label(k)==i)
				{
					train_labels->set_label(k,1.0);
					subset_labels[tot] = k;
					subset_feats[tot]  = k;
					tot++;
				}
				else if (m_labels->get_int_label(k)==j)
				{
					train_labels->set_label(k,-1.0);
					subset_labels[tot] = k;
					subset_feats[tot]  = k;
					tot++;
				}
			}

			train_labels->set_subset( new CSubset( SGVector<index_t>(subset_labels.vector, tot) ) );
			set_machine_subset( new CSubset( SGVector<index_t>(subset_feats.vector, tot) ) );

			m_machine->train();
			m_machines.push_back(get_machine_from_trained(m_machine));

			train_labels->remove_subset();
			remove_machine_subset();
		}
	}

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
			CMachine *machine = m_machines.get_element(i);
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
			CMachine *machine = m_machines.get_element(i);
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
	SG_NOTIMPLEMENTED;
	return 0;
}
