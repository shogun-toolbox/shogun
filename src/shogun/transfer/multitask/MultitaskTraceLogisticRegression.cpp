/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <transfer/multitask/MultitaskTraceLogisticRegression.h>
#include <lib/malsar/malsar_low_rank.h>
#include <lib/malsar/malsar_options.h>
#include <lib/IndexBlockGroup.h>
#include <lib/SGVector.h>

namespace shogun
{

CMultitaskTraceLogisticRegression::CMultitaskTraceLogisticRegression() :
	CMultitaskLogisticRegression(), m_rho(0.0)
{
	init();
}

CMultitaskTraceLogisticRegression::CMultitaskTraceLogisticRegression(
     float64_t rho, CDotFeatures* train_features,
     CBinaryLabels* train_labels, CTaskGroup* task_group) :
	CMultitaskLogisticRegression(0.0,train_features,train_labels,(CTaskRelation*)task_group)
{
	set_rho(rho);
	init();
}

void CMultitaskTraceLogisticRegression::init()
{
	SG_ADD(&m_rho,"rho","rho",MS_AVAILABLE);
}

void CMultitaskTraceLogisticRegression::set_rho(float64_t rho)
{
	m_rho = rho;
}

float64_t CMultitaskTraceLogisticRegression::get_rho() const
{
	return m_rho;
}

CMultitaskTraceLogisticRegression::~CMultitaskTraceLogisticRegression()
{
}

bool CMultitaskTraceLogisticRegression::train_locked_implementation(SGVector<index_t>* tasks)
{
	SGVector<float64_t> y(m_labels->get_num_labels());
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = ((CBinaryLabels*)m_labels)->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
	options.tasks_indices = tasks;

#ifdef HAVE_EIGEN3
	malsar_result_t model = malsar_low_rank(
		features, y.vector, m_rho, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Please install Eigen3 to use MultitaskTraceLogisticRegression\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif
	return true;
}

bool CMultitaskTraceLogisticRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features)
	ASSERT(m_labels)
	ASSERT(m_task_relation)

	SGVector<float64_t> y(m_labels->get_num_labels());
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = ((CBinaryLabels*)m_labels)->get_label(i);

	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
	options.tasks_indices = ((CTaskGroup*)m_task_relation)->get_tasks_indices();

#ifdef HAVE_EIGEN3
	malsar_result_t model = malsar_low_rank(
		features, y.vector, m_rho, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Please install Eigen3 to use MultitaskTraceLogisticRegression\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif

	SG_FREE(options.tasks_indices);

	return true;
}

}
