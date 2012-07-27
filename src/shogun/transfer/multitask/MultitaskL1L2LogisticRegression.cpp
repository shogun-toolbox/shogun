/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskL1L2LogisticRegression.h>
#include <shogun/lib/malsar/malsar_joint_feature_learning.h>
#include <shogun/lib/malsar/malsar_options.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

CMultitaskL1L2LogisticRegression::CMultitaskL1L2LogisticRegression() :
	CMultitaskLogisticRegression(), m_rho1(0.0), m_rho2(0.0)
{
}

CMultitaskL1L2LogisticRegression::CMultitaskL1L2LogisticRegression(
     float64_t rho1, float64_t rho2, CDotFeatures* train_features, 
     CBinaryLabels* train_labels, CTaskGroup* task_group) :
	CMultitaskLogisticRegression(0.0,train_features,train_labels,(CTaskRelation*)task_group)
{
	set_rho1(rho1);
	set_rho2(rho2);
}

void CMultitaskL1L2LogisticRegression::set_rho1(float64_t rho1)
{
	m_rho1 = rho1;
}

void CMultitaskL1L2LogisticRegression::set_rho2(float64_t rho2)
{
	m_rho2 = rho2;
}

CMultitaskL1L2LogisticRegression::~CMultitaskL1L2LogisticRegression()
{
}

bool CMultitaskL1L2LogisticRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features);
	ASSERT(m_labels);
	ASSERT(m_task_relation);

	SGVector<float64_t> y(m_labels->get_num_labels());
	for (int32_t i=0; i<y.vlen; i++)
		y[i] = ((CBinaryLabels*)m_labels)->get_label(i);
	
	malsar_options options = malsar_options::default_options();
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.n_tasks = ((CTaskGroup*)m_task_relation)->get_num_tasks();
	SGVector<index_t>* subset_tasks_indices = get_subset_tasks_indices();
	options.tasks_indices = subset_tasks_indices;

#ifdef HAVE_EIGEN3
	malsar_result_t model = malsar_joint_feature_learning(
		features, y.vector, m_rho1, m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Please install Eigen3 to use MultitaskL1L2LogisticRegression\n");
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks); 
	m_tasks_c = SGVector<float64_t>(options.n_tasks); 
#endif

	for (int32_t i=0; i<options.n_tasks; i++)
		subset_tasks_indices[i].~SGVector<index_t>();
	SG_FREE(subset_tasks_indices);
	
	return true;
}

}
