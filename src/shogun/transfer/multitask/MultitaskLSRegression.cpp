/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLSRegression.h>
#include <shogun/lib/slep/slep_tree_mt_lsr.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CMultitaskLSRegression::CMultitaskLSRegression() :
	CSLEPMachine(), m_task_relation(NULL)
{
	register_parameters();
}

CMultitaskLSRegression::CMultitaskLSRegression(
     float64_t z, CDotFeatures* train_features, 
     CRegressionLabels* train_labels, CTaskRelation* task_relation) :
	CSLEPMachine(z,train_features,(CLabels*)train_labels), m_task_relation(NULL)
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
	ASSERT(task>0);
	ASSERT(task<m_tasks_w.num_cols);
	m_current_task = task;
	int32_t n_feats = ((CDotFeatures*)features)->get_dim_feature_space();
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

	int32_t n_vectors = features->get_num_vectors();

	float64_t* y = SG_MALLOC(float64_t, n_vectors);
	for (int32_t i=0; i<n_vectors; i++)
		y[i] = ((CRegressionLabels*)m_labels)->get_label(i);

	slep_options options;

	SG_NOTIMPLEMENTED;

	SG_FREE(y);

	return true;
}

}
