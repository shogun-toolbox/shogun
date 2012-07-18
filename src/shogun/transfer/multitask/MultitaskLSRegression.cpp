/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLSRegression.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/lib/slep/slep_solver.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CMultitaskLSRegression::CMultitaskLSRegression() :
	CSLEPMachine(), m_current_task(0), 
	m_task_relation(NULL)
{
	register_parameters();
}

CMultitaskLSRegression::CMultitaskLSRegression(
     float64_t z, CDotFeatures* train_features, 
     CRegressionLabels* train_labels, CTaskRelation* task_relation) :
	CSLEPMachine(z,train_features,(CLabels*)train_labels), 
	m_current_task(0), m_task_relation(NULL)
{
	set_task_relation(task_relation);
	register_parameters();
}

CMultitaskLSRegression::~CMultitaskLSRegression()
{
	SG_UNREF(m_task_relation);
}

void CMultitaskLSRegression::register_parameters()
{
	SG_ADD((CSGObject**)&m_task_relation, "task_relation", "task relation", MS_NOT_AVAILABLE);
}

int32_t CMultitaskLSRegression::get_current_task() const
{
	return m_current_task;
}

void CMultitaskLSRegression::set_current_task(int32_t task)
{
	ASSERT(task>=0);
	ASSERT(task<m_tasks_w.num_cols);
	m_current_task = task;
	int32_t n_feats = m_tasks_w.num_rows;
	w = SGVector<float64_t>(n_feats);
	for (int32_t i=0; i<n_feats; i++)
		w[i] = m_tasks_w(i,task);
}

CTaskRelation* CMultitaskLSRegression::get_task_relation() const
{
	SG_REF(m_task_relation);
	return m_task_relation;
}

void CMultitaskLSRegression::set_task_relation(CTaskRelation* task_relation)
{
	SG_UNREF(m_task_relation);
	SG_REF(task_relation);
	m_task_relation = task_relation;
}

bool CMultitaskLSRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);

	SGVector<float64_t> y = ((CRegressionLabels*)m_labels)->get_labels();
	
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
			options.mode = MULTITASK_GROUP;
			options.loss = LEAST_SQUARES;
			m_tasks_w = slep_solver(features, y.vector, m_z, options).w;
		}
		break;
		case TASK_TREE: 
		{
			CTaskTree* task_tree = (CTaskTree*)m_task_relation;
			SGVector<index_t> ind = task_tree->get_SLEP_ind();
			options.ind = ind.vector;
			SGVector<float64_t> ind_t = task_tree->get_SLEP_ind_t();
			options.ind_t = ind_t.vector;
			options.n_tasks = ind.vlen-1;
			options.n_nodes = ind_t.vlen/3;
			options.mode = MULTITASK_TREE;
			options.loss = LEAST_SQUARES;
			m_tasks_w = slep_solver(features, y.vector, m_z, options).w;
		}
		break;
		default: 
			SG_ERROR("Not supported task relation type\n");
	}

	return true;
}

}
