/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/TraceNormMultitaskLSRegression.h>
#include <shogun/lib/slep/slep_accel_grad_mtl.h>

namespace shogun
{

CTraceNormMultitaskLSRegression::CTraceNormMultitaskLSRegression() :
	CMultitaskLSRegression()
{
}

CTraceNormMultitaskLSRegression::CTraceNormMultitaskLSRegression(
     float64_t z, CDotFeatures* train_features, 
     CRegressionLabels* train_labels, CIndexBlockGroup* task_group) :
	CMultitaskLSRegression(z,train_features,train_labels,(CIndexBlockRelation*)task_group) 
{
}

CTraceNormMultitaskLSRegression::~CTraceNormMultitaskLSRegression()
{
}

bool CTraceNormMultitaskLSRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);
	ASSERT(m_task_relation->get_relation_type()==GROUP);

	SGVector<float64_t> y = ((CRegressionLabels*)m_labels)->get_labels();
	
	slep_options options = slep_options::default_options();
	options.tolerance = m_tolerance;

	CIndexBlockGroup* task_group = (CIndexBlockGroup*)m_task_relation;
	SGVector<index_t> ind = task_group->get_SLEP_ind();
	options.ind = ind.vector;
	options.n_nodes = ind.vlen-1;

	m_tasks_w = slep_accel_grad_mtl(features, y.vector, m_z, options);

	return true;
}

}
