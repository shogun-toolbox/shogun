/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskCompositeMachine.h>

#include <map>
#include <vector>

using namespace std;

namespace shogun
{

CMultitaskCompositeMachine::CMultitaskCompositeMachine() :
	CMachine(), m_machine(NULL), m_features(NULL), m_current_task(0), 
	m_task_group(NULL)
{
	register_parameters();
}

CMultitaskCompositeMachine::CMultitaskCompositeMachine(
     CMachine* machine, CFeatures* train_features, 
     CLabels* train_labels, CTaskGroup* task_group) :
	CMachine(), m_machine(NULL), m_features(NULL), 
	m_current_task(0), m_task_group(NULL)
{
	set_machine(machine);
	set_features(train_features);
	set_labels(train_labels);
	set_task_group(task_group);
	register_parameters();
}

CMultitaskCompositeMachine::~CMultitaskCompositeMachine()
{
	SG_UNREF(m_machine);
	SG_UNREF(m_features);
	SG_UNREF(m_task_machines);
	SG_UNREF(m_task_group);
}

void CMultitaskCompositeMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_machine, "machine", "machine", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_features, "features", "features", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_task_machines, "task_machines", "task machines", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_task_group, "task_group", "task group", MS_NOT_AVAILABLE);
}

int32_t CMultitaskCompositeMachine::get_current_task() const
{
	return m_current_task;
}

void CMultitaskCompositeMachine::set_current_task(int32_t task)
{
	m_current_task = task;
}

CTaskGroup* CMultitaskCompositeMachine::get_task_group() const
{
	SG_REF(m_task_group);
	return m_task_group;
}

void CMultitaskCompositeMachine::set_task_group(CTaskGroup* task_group)
{
	SG_UNREF(m_task_group);
	SG_REF(task_group);
	m_task_group = task_group;
}

bool CMultitaskCompositeMachine::train_machine(CFeatures* data)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CMultitaskCompositeMachine::post_lock(CLabels* labels, CFeatures* features)
{
	ASSERT(m_task_group)
	set_features(m_features);
	if (!m_machine->is_data_locked())
		m_machine->data_lock(labels,features);

	int n_tasks = m_task_group->get_num_tasks();
	SGVector<index_t>* tasks_indices = m_task_group->get_tasks_indices();

	m_tasks_indices.clear();
	for (int32_t i=0; i<n_tasks; i++)
	{
		set<index_t> indices_set;
		SGVector<index_t> task_indices = tasks_indices[i];
		for (int32_t j=0; j<task_indices.vlen; j++)
			indices_set.insert(task_indices[j]);

		m_tasks_indices.push_back(indices_set);
	}

	SG_FREE(tasks_indices);
}

bool CMultitaskCompositeMachine::train_locked(SGVector<index_t> indices)
{
	int n_tasks = m_task_group->get_num_tasks();
	ASSERT((int)m_tasks_indices.size()==n_tasks)
	vector< vector<index_t> > cutted_task_indices;
	for (int32_t i=0; i<n_tasks; i++)
		cutted_task_indices.push_back(vector<index_t>());
	for (int32_t i=0; i<indices.vlen; i++)
	{
		for (int32_t j=0; j<n_tasks; j++)
		{
			if (m_tasks_indices[j].count(indices[i]))
			{
				cutted_task_indices[j].push_back(indices[i]);
				break;
			}
		}
	}
	//SG_UNREF(m_task_machines);
	m_task_machines = new CDynamicObjectArray();
	for (int32_t i=0; i<n_tasks; i++)
	{
		SGVector<index_t> task_indices(cutted_task_indices[i].size());
		for (int32_t j=0; j<(int)cutted_task_indices[i].size(); j++)
			task_indices[j] = cutted_task_indices[i][j];

		m_machine->train_locked(task_indices);
		m_task_machines->push_back(m_machine->clone());
	}
	return true;
}

float64_t CMultitaskCompositeMachine::apply_one(int32_t i)
{
	CMachine* m = (CMachine*)(m_task_machines->get_element(m_current_task));
	float64_t result = m->apply_one(i);
	SG_UNREF(m);
	return result;
}

CBinaryLabels* CMultitaskCompositeMachine::apply_locked_binary(SGVector<index_t> indices)
{
	int n_tasks = m_task_group->get_num_tasks();
	SGVector<float64_t> result(indices.vlen);
	result.zero();
	for (int32_t i=0; i<indices.vlen; i++)
	{
		for (int32_t j=0; j<n_tasks; j++)
		{
			if (m_tasks_indices[j].count(indices[i]))
			{
				set_current_task(j);
				result[i] = apply_one(indices[i]);
				break;
			}
		}
	}
	return new CBinaryLabels(result);
}

}
