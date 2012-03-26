/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2012 Soeren Sonnenburg and Sergey Lisitsyn
 * One vs. One strategy written (W) 2012 Fernando José Iglesias García and Sergey Lisitsyn
 * Copyright (C) 1999-2012 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/machine/MulticlassMachine.h>
#include <shogun/base/Parameter.h>
#include <shogun/features/Labels.h>

using namespace shogun;

CMulticlassMachine::CMulticlassMachine()
: CMachine(), m_multiclass_strategy(ONE_VS_REST_STRATEGY), m_rejection_strategy(NULL)
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
	m_parameters->add((machine_int_t*)&m_multiclass_strategy,"m_multiclass_type");
	m_parameters->add((CSGObject**)&m_machine, "m_machine");
	m_parameters->add((CSGObject**)&m_rejection_strategy, "m_rejection_strategy");
	m_parameters->add_vector((CSGObject***)&m_machines.vector,&m_machines.vlen, "m_machines");
}

void CMulticlassMachine::clear_machines()
{
	for(int32_t i=0; i<m_machines.vlen; i++)
		SG_UNREF(m_machines[i]);

	m_machines.destroy_vector();
}

CLabels* CMulticlassMachine::apply(CFeatures* features)
{
	init_machines_for_apply(features);
	return apply();
}

CLabels* CMulticlassMachine::apply()
{
	switch (m_multiclass_strategy)
	{
			case ONE_VS_REST_STRATEGY:
				return classify_one_vs_rest();
				break;
			case ONE_VS_ONE_STRATEGY:
				return classify_one_vs_one();
				break;
			default:
				SG_ERROR("Unknown multiclass strategy\n");
	}
	return NULL;
}

bool CMulticlassMachine::train_machine(CFeatures* data)
{
	if (!data && !is_ready())
		SG_ERROR("Please provide training data.\n");

	if (data)
	{
		init_machine_for_train(data);
		init_machines_for_apply(data);
	}

	switch (m_multiclass_strategy)
	{
			case ONE_VS_REST_STRATEGY:
				return train_one_vs_rest();
				break;
			case ONE_VS_ONE_STRATEGY:
				return train_one_vs_one();
				break;
			default:
				SG_ERROR("Unknown multiclass strategy\n");
	}

	return NULL;
}

bool CMulticlassMachine::train_one_vs_rest()
{
	int32_t num_classes = m_labels->get_num_classes();
	int32_t num_vectors = get_num_rhs_vectors();

	clear_machines();
	m_machines = SGVector<CMachine*>(num_classes);
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
		m_machines[i] = get_machine_from_trained(m_machine);
	}

	SG_UNREF(train_labels);
	return true;
}

bool CMulticlassMachine::train_one_vs_one()
{
	int32_t num_classes = m_labels->get_num_classes();
	int32_t num_vectors = get_num_rhs_vectors();

	clear_machines();
	m_machines = SGVector<CMachine*>(num_classes*(num_classes-1)/2);
	CLabels* train_labels = new CLabels(num_vectors);
	SG_REF(train_labels);
	m_machine->set_labels(train_labels);

	/** Number of vectors included in every subset */
	int32_t tot = 0;

	for (int32_t i=0, c=0; i<num_classes; i++)
	{
		for (int32_t j=i+1; j<num_classes; j++)
		{
			/** TODO fix this, there must be a way not to allocate these guys on every
			 * iteration */
			SGVector<index_t> subset_labels(num_vectors);
			SGVector<index_t> subset_feats(num_vectors);

			tot = 0;
			for (int32_t k=0; k<num_vectors; k++)
			{
				if (m_labels->get_int_label(k)==i)
				{
					train_labels->set_label(k,-1.0);
					subset_labels[tot] = k;
					subset_feats[tot]  = k;
					tot++;
				}
				else if (m_labels->get_int_label(k)==j)
				{
					train_labels->set_label(k,1.0);
					subset_labels[tot] = k;
					subset_feats[tot]  = k;
					tot++;
				}
			}

			train_labels->set_subset( new CSubset( SGVector<index_t>(subset_labels.vector, tot) ) );
			set_machine_subset( new CSubset( SGVector<index_t>(subset_feats.vector, tot) ) );

			m_machine->train();
			m_machines[c++] = get_machine_from_trained(m_machine);

			train_labels->remove_subset();
			remove_machine_subset();
		}
	}

	SG_PRINT(">>>> at the end of training num_machines = %d\n", get_num_machines());

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
			ASSERT(m_machines[i]);
			outputs[i]=m_machines[i]->apply();
		}

		SGVector<float64_t> outputs_for_i(num_machines);
		for (int32_t i=0; i<num_vectors; i++)
		{
			int32_t winner = 0;

			for (int32_t j=0; j<num_machines; j++)
				outputs_for_i[j] = outputs[j]->get_label(i);

			if (m_rejection_strategy && m_rejection_strategy->reject(outputs_for_i))
			{
				winner=result->REJECTION_LABEL;
			}
			else
			{
				float64_t max_out = outputs[0]->get_label(i);

				for (int32_t j=1; j<num_machines; j++)
				{
					if (outputs_for_i[j]>max_out)
					{
						max_out = outputs_for_i[j];
						winner = j;
					}
				}
			}
			result->set_label(i, winner);
		}
		outputs_for_i.destroy_vector();

		for (int32_t i=0; i<num_machines; i++)
			SG_UNREF(outputs[i]);

		SG_FREE(outputs);
	}

	return result;
}

CLabels* CMulticlassMachine::classify_one_vs_one()
{
	int32_t num_classes  = m_labels->get_num_classes();
	int32_t num_machines = get_num_machines();
	SG_PRINT(">>>> at the beginning of classify num_machines = %d\n", num_machines);
	SG_PRINT(">>>> at the beginning of classify num_classes = %d\n",  num_classes);
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
			ASSERT(m_machines[i]);
			outputs[i]=m_machines[i]->apply();
		}

		SG_PRINT(">>>> Starting to go through the vectors\n");

		SGVector<float64_t> votes(num_classes);
		for (int32_t v=0; v<num_vectors; v++)
		{
			int32_t s=0;
			votes.zero();

			for (int32_t i=0; i<num_classes; i++)
			{
				for (int32_t j=i+1; j<num_classes; j++)
				{
					SG_PRINT(">>>> s = %d v  = %d\n", s, v);
					if ( ! outputs[s] )
						SG_ERROR(">>>>>> outputs[%d] is null!!!\n", s);
					if (outputs[s++]->get_label(v)>0)
						votes[i]++;
					else
						votes[j]++;
				}
			}

			SG_PRINT(">>>> votes counted\n");

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

			SG_PRINT(">>>> max_votes found\n");
			result->set_label(v, winner);
		}

		SG_PRINT(">>>> Destroy votes\n");
		votes.destroy_vector();

		for (int32_t i=0; i<num_machines; i++)
			SG_UNREF(outputs[i]);
		SG_FREE(outputs);
	}

	return result;
}

float64_t CMulticlassMachine::apply(int32_t num)
{
	SG_NOTIMPLEMENTED;
	return 0;
}
