/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskL12LogisticRegression.h>
#include <shogun/lib/malsar/malsar_joint_feature_learning.h>
#include <shogun/lib/malsar/malsar_options.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{

CMultitaskL12LogisticRegression::CMultitaskL12LogisticRegression() :
	CMultitaskLogisticRegression(), m_rho1(0.0), m_rho2(0.0)
{
	init();
}

CMultitaskL12LogisticRegression::CMultitaskL12LogisticRegression(
     float64_t rho1, float64_t rho2, CDotFeatures* train_features,
     CBinaryLabels* train_labels, CTaskGroup* task_group) :
	CMultitaskLogisticRegression(0.0,train_features,train_labels,(CTaskRelation*)task_group)
{
	set_rho1(rho1);
	set_rho2(rho2);
	init();
}

void CMultitaskL12LogisticRegression::init()
{
	SG_ADD(&m_rho1,"rho1","rho L1/L2 regularization parameter",MS_AVAILABLE);
	SG_ADD(&m_rho2,"rho2","rho L2 regularization parameter",MS_AVAILABLE);
}

void CMultitaskL12LogisticRegression::set_rho1(float64_t rho1)
{
	m_rho1 = rho1;
}

void CMultitaskL12LogisticRegression::set_rho2(float64_t rho2)
{
	m_rho2 = rho2;
}

float64_t CMultitaskL12LogisticRegression::get_rho1() const
{
	return m_rho1;
}

float64_t CMultitaskL12LogisticRegression::get_rho2() const
{
	return m_rho2;
}

CMultitaskL12LogisticRegression::~CMultitaskL12LogisticRegression()
{
}

bool CMultitaskL12LogisticRegression::train_locked_implementation(SGVector<index_t>* tasks)
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
	malsar_result_t model = malsar_joint_feature_learning(
		features, y.vector, m_rho1, m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Please install Eigen3 to use MultitaskL12LogisticRegression\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif

	return true;
}

bool CMultitaskL12LogisticRegression::train_machine(CFeatures* data)
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
	malsar_result_t model = malsar_joint_feature_learning(
		features, y.vector, m_rho1, m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Please install Eigen3 to use MultitaskL12LogisticRegression\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif

	SG_FREE(options.tasks_indices);

	return true;
}

}
