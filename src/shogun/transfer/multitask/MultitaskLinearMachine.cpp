/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLinearMachine.h>
#include <shogun/lib/slep/slep_solver.h>
#include <shogun/lib/slep/slep_options.h>

#include <map>
#include <vector>

using namespace std;

namespace shogun
{

CMultitaskLinearMachine::CMultitaskLinearMachine() :
	CLinearMachine(), m_current_task(0), 
	m_task_relation(NULL)
{
	register_parameters();
}

CMultitaskLinearMachine::CMultitaskLinearMachine(
     CDotFeatures* train_features, 
     CLabels* train_labels, CTaskRelation* task_relation) :
	CLinearMachine(), m_current_task(0), m_task_relation(NULL)
{
	set_features(train_features);
	set_labels(train_labels);
	set_task_relation(task_relation);
	register_parameters();
}

CMultitaskLinearMachine::~CMultitaskLinearMachine()
{
	SG_UNREF(m_task_relation);
}

void CMultitaskLinearMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_task_relation, "task_relation", "task relation", MS_NOT_AVAILABLE);
}

int32_t CMultitaskLinearMachine::get_current_task() const
{
	return m_current_task;
}

void CMultitaskLinearMachine::set_current_task(int32_t task)
{
	ASSERT(task>=0)
	ASSERT(task<m_tasks_w.num_cols)
	m_current_task = task;
}

CTaskRelation* CMultitaskLinearMachine::get_task_relation() const
{
	SG_REF(m_task_relation);
	return m_task_relation;
}

void CMultitaskLinearMachine::set_task_relation(CTaskRelation* task_relation)
{
	SG_REF(task_relation);
	SG_UNREF(m_task_relation);
	m_task_relation = task_relation;
}

bool CMultitaskLinearMachine::train_machine(CFeatures* data)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CMultitaskLinearMachine::post_lock(CLabels* labels, CFeatures* features_)
{
	set_features((CDotFeatures*)features_);
	int n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
	SGVector<index_t>* tasks_indices = ((CTaskGroup*)m_task_relation)->get_tasks_indices();

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

bool CMultitaskLinearMachine::train_locked(SGVector<index_t> indices)
{
	int n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
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
	SGVector<index_t>* tasks = SG_MALLOC(SGVector<index_t>, n_tasks);
	for (int32_t i=0; i<n_tasks; i++)
	{
		tasks[i]=SGVector<index_t>(cutted_task_indices[i].size());
		for (int32_t j=0; j<(int)cutted_task_indices[i].size(); j++)
			tasks[i][j] = cutted_task_indices[i][j];
		//tasks[i].display_vector();
	}
	bool res = train_locked_implementation(tasks);
	SG_FREE(tasks);
	return res;
}

bool CMultitaskLinearMachine::train_locked_implementation(SGVector<index_t>* tasks)
{
	SG_NOTIMPLEMENTED
	return false;
}

CBinaryLabels* CMultitaskLinearMachine::apply_locked_binary(SGVector<index_t> indices)
{
	int n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
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

float64_t CMultitaskLinearMachine::apply_one(int32_t i)
{
	SG_NOTIMPLEMENTED
	return 0.0;
}

SGVector<float64_t> CMultitaskLinearMachine::apply_get_outputs(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")

		set_features((CDotFeatures*) data);
	}

	if (!features)
		return SGVector<float64_t>();

	int32_t num=features->get_num_vectors();
	ASSERT(num>0)
	float64_t* out=SG_MALLOC(float64_t, num);
	for (int32_t i=0; i<num; i++)
		out[i] = apply_one(i);

	return SGVector<float64_t>(out,num);
}

SGVector<float64_t> CMultitaskLinearMachine::get_w() const
{
	SGVector<float64_t> w_(m_tasks_w.num_rows);
	for (int32_t i=0; i<w_.vlen; i++)
		w_[i] = m_tasks_w(i,m_current_task);
	return w_;
}

void CMultitaskLinearMachine::set_w(const SGVector<float64_t> src_w)
{
	for (int32_t i=0; i<m_tasks_w.num_rows; i++)
		m_tasks_w(i,m_current_task) = src_w[i];
}

void CMultitaskLinearMachine::set_bias(float64_t b)
{
	m_tasks_c[m_current_task] = b;
}

float64_t CMultitaskLinearMachine::get_bias()
{
	return m_tasks_c[m_current_task];
}

SGVector<index_t>* CMultitaskLinearMachine::get_subset_tasks_indices()
{
	int n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
	SGVector<index_t>* tasks_indices = ((CTaskGroup*)m_task_relation)->get_tasks_indices();

	CSubsetStack* sstack = features->get_subset_stack();
	map<index_t,index_t> subset_inv_map = map<index_t,index_t>();
	for (int32_t i=0; i<sstack->get_size(); i++)
		subset_inv_map[sstack->subset_idx_conversion(i)] = i;

	SGVector<index_t>* subset_tasks_indices = SG_MALLOC(SGVector<index_t>, n_tasks);
	for (int32_t i=0; i<n_tasks; i++)
	{
		SGVector<index_t> task = tasks_indices[i];
		//task.display_vector("task");
		vector<index_t> cutted = vector<index_t>();
		for (int32_t j=0; j<task.vlen; j++)
		{
			if (subset_inv_map.count(task[j]))
				cutted.push_back(subset_inv_map[task[j]]);
		}
		SGVector<index_t> cutted_task(cutted.size());
		for (int32_t j=0; j<cutted_task.vlen; j++)
			cutted_task[j] = cutted[j];
		//cutted_task.display_vector("cutted");
		subset_tasks_indices[i] = cutted_task;
	}
	SG_FREE(tasks_indices);
	
	return subset_tasks_indices;
}


}
