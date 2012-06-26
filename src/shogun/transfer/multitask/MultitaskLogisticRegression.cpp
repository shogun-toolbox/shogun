/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
#include <shogun/lib/slep/slep_mt_lr.h>
#include <shogun/lib/slep/slep_options.h>

#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>

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
		case GROUP:
		{
			CTaskGroup* task_group = (CTaskGroup*)m_task_relation;
			SGVector<index_t> ind = task_group->get_SLEP_ind();
			options.ind = ind.vector;
			options.n_tasks = ind.vlen-1;

			slep_result_t result = slep_mt_lr(features, y.vector, m_z, options);
			m_tasks_w = result.w;
			m_tasks_c = result.c;
		}
		break;
		case TREE: 
		{
			SG_ERROR("Not supported task relation type\n");
		}
		default: 
			SG_ERROR("Not supported task relation type\n");
	}

	return true;
}

}
