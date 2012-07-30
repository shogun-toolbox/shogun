/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
#include <shogun/lib/slep/slep_solver.h>
#include <shogun/lib/slep/slep_options.h>

#include <map>
#include <vector>

using namespace std;

namespace shogun
{

CMultitaskLogisticRegression::CMultitaskLogisticRegression() :
	CSLEPMachine(), m_current_task(0), 
	m_task_relation(NULL)
{
	register_parameters();
}

CMultitaskLogisticRegression::CMultitaskLogisticRegression(
     float64_t z, CDotFeatures* train_features, 
     CBinaryLabels* train_labels, CTaskRelation* task_relation) :
	CSLEPMachine(z,train_features,(CLabels*)train_labels), 
	m_current_task(0), m_task_relation(NULL)
{
	set_task_relation(task_relation);
	register_parameters();
}

CMultitaskLogisticRegression::~CMultitaskLogisticRegression()
{
	SG_UNREF(m_task_relation);
}

void CMultitaskLogisticRegression::register_parameters()
{
	SG_ADD((CSGObject**)&m_task_relation, "task_relation", "task relation", MS_NOT_AVAILABLE);
}

int32_t CMultitaskLogisticRegression::get_current_task() const
{
	return m_current_task;
}

void CMultitaskLogisticRegression::set_current_task(int32_t task)
{
	ASSERT(task>=0);
	ASSERT(task<m_tasks_w.num_cols);
	m_current_task = task;
	int32_t n_feats = m_tasks_w.num_rows;
	w = SGVector<float64_t>(n_feats);
	for (int32_t i=0; i<n_feats; i++)
		w[i] = m_tasks_w(i,task);

	bias = m_tasks_c[task];
}

CTaskRelation* CMultitaskLogisticRegression::get_task_relation() const
{
	SG_REF(m_task_relation);
	return m_task_relation;
}

void CMultitaskLogisticRegression::set_task_relation(CTaskRelation* task_relation)
{
	SG_UNREF(m_task_relation);
	SG_REF(task_relation);
	m_task_relation = task_relation;
}

bool CMultitaskLogisticRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);

	SGVector<float64_t> y = ((CBinaryLabels*)m_labels)->get_labels();
	
	slep_options options = slep_options::default_options();
	options.q = m_q;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;

	ETaskRelationType relation_type = m_task_relation->get_relation_type();
	switch (relation_type)
	{
		case TASK_GROUP:
		{
			CTaskGroup* task_group = (CTaskGroup*)m_task_relation;
			SGVector<index_t> ind = task_group->get_SLEP_ind();
			options.ind = ind.vector;
			options.n_tasks = ind.vlen-1;
			if (ind[ind.vlen-1] > features->get_num_vectors())
				SG_ERROR("Group of tasks covers more vectors than available\n");
			
			options.mode = MULTITASK_GROUP;
			options.loss = LOGISTIC;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		case TASK_TREE: 
		{
			CTaskTree* task_tree = (CTaskTree*)m_task_relation;

			CTask* root_task = (CTask*)task_tree->get_root_task();
			//if (root_task->get_max_index() > features->get_num_vectors())
			//	SG_ERROR("Root task covers more vectors than available\n");
			SG_UNREF(root_task);

			SGVector<index_t> ind = task_tree->get_SLEP_ind();
			SGVector<float64_t> ind_t = task_tree->get_SLEP_ind_t();
			options.ind = ind.vector;
			options.ind_t = ind_t.vector;
			options.n_tasks = ind.vlen-1;
			options.n_nodes = ind_t.vlen / 3;
			options.mode = MULTITASK_TREE;
			options.loss = LOGISTIC;
			slep_result_t result = slep_solver(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		default: 
			SG_ERROR("Not supported task relation type\n");
	}

	return true;
}

void CMultitaskLogisticRegression::post_lock()
{
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

	for (int32_t i=0; i<n_tasks; i++)
		tasks_indices[i].~SGVector<index_t>();
	SG_FREE(tasks_indices);
}

bool CMultitaskLogisticRegression::train_locked(SGVector<index_t> indices)
{
	int n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
	ASSERT((int)m_tasks_indices.size()==n_tasks);
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
		new (&tasks[i]) SGVector<index_t>(cutted_task_indices[i].size());
		for (int32_t j=0; j<(int)cutted_task_indices[i].size(); j++)
			tasks[i][j] = cutted_task_indices[i][j];
		//tasks[i].display_vector();
	}
	bool res = train_locked_implementation(indices,tasks);
	for (int32_t i=0; i<n_tasks; i++)
		tasks[i].~SGVector<index_t>();
	SG_FREE(tasks);
	return res;
}

bool CMultitaskLogisticRegression::train_locked_implementation(SGVector<index_t> indices,
                                                               SGVector<index_t>* tasks)
{
	SG_NOTIMPLEMENTED;
	return false;
}

CBinaryLabels* CMultitaskLogisticRegression::apply_locked_binary(SGVector<index_t> indices)
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

SGVector<index_t>* CMultitaskLogisticRegression::get_subset_tasks_indices()
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
		new (&subset_tasks_indices[i]) SGVector<index_t>();
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
	for (int32_t i=0; i<n_tasks; i++)
		tasks_indices[i].~SGVector<index_t>();
	SG_FREE(tasks_indices);
	
	return subset_tasks_indices;
}


}
