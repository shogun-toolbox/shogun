/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskTraceLogisticRegression.h>
#include <shogun/lib/slep/malsar_low_rank.h>
#include <shogun/lib/slep/slep_options.h>
#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

CMultitaskTraceLogisticRegression::CMultitaskTraceLogisticRegression() :
	CMultitaskLogisticRegression(), m_rho(0.0)
{
}

CMultitaskTraceLogisticRegression::CMultitaskTraceLogisticRegression(
     float64_t rho, CDotFeatures* train_features, 
     CBinaryLabels* train_labels, CIndexBlockRelation* task_relation) :
	CMultitaskLogisticRegression(0.0,train_features,train_labels,task_relation)
{
	set_rho(rho);
}

void CMultitaskTraceLogisticRegression::set_rho(float64_t rho)
{
	m_rho = rho;
}

CMultitaskTraceLogisticRegression::~CMultitaskTraceLogisticRegression()
{
}

bool CMultitaskTraceLogisticRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);

	SGVector<float64_t> y(m_labels->get_num_labels());
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = ((CBinaryLabels*)m_labels)->get_label(i);
	
	slep_options options = slep_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	SGVector<index_t> ind = ((CIndexBlockGroup*)m_task_relation)->get_SLEP_ind();
	options.ind = ind.vector;
	options.n_tasks = ind.vlen-1;

#ifdef HAVE_EIGEN3
	slep_result_t model = malsar_low_rank(
		features, y.vector, m_rho, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#endif

	ASSERT(m_task_relation);

	return true;
}

}
