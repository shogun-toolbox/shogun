/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskClusteredLogisticRegression.h>
#include <shogun/lib/malsar/malsar_clustered.h>
#include <shogun/lib/malsar/malsar_options.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

CMultitaskClusteredLogisticRegression::CMultitaskClusteredLogisticRegression() :
	CMultitaskLogisticRegression(), m_rho1(0.0), m_rho2(0.0)
{
}

CMultitaskClusteredLogisticRegression::CMultitaskClusteredLogisticRegression(
     float64_t rho1, float64_t rho2, CDotFeatures* train_features,
     CBinaryLabels* train_labels, CTaskGroup* task_group, int32_t n_clusters) :
	CMultitaskLogisticRegression(0.0,train_features,train_labels,(CTaskRelation*)task_group)
{
	set_rho1(rho1);
	set_rho2(rho2);
	set_num_clusters(n_clusters);
}

int32_t CMultitaskClusteredLogisticRegression::get_rho1() const
{
	return m_rho1;
}

int32_t CMultitaskClusteredLogisticRegression::get_rho2() const
{
	return m_rho2;
}

void CMultitaskClusteredLogisticRegression::set_rho1(float64_t rho1)
{
	m_rho1 = rho1;
}

void CMultitaskClusteredLogisticRegression::set_rho2(float64_t rho2)
{
	m_rho2 = rho2;
}

int32_t CMultitaskClusteredLogisticRegression::get_num_clusters() const
{
	return m_num_clusters;
}

void CMultitaskClusteredLogisticRegression::set_num_clusters(int32_t num_clusters)
{
	m_num_clusters = num_clusters;
}

CMultitaskClusteredLogisticRegression::~CMultitaskClusteredLogisticRegression()
{
}

bool CMultitaskClusteredLogisticRegression::train_locked_implementation(SGVector<index_t>* tasks)
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
	options.n_clusters = m_num_clusters;

#ifdef HAVE_EIGEN3
#ifndef HAVE_CXX11
	malsar_result_t model = malsar_clustered(
		features, y.vector, m_rho1, m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Clustered LR is unstable with C++11\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_w.set_const(0);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
	m_tasks_c.set_const(0);
#endif
#else
	SG_WARNING("Please install Eigen3 to use MultitaskClusteredLogisticRegression\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif
	return true;
}

bool CMultitaskClusteredLogisticRegression::train_machine(CFeatures* data)
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
	options.n_clusters = m_num_clusters;

#ifdef HAVE_EIGEN3
#ifndef HAVE_CXX11
	malsar_result_t model = malsar_clustered(
		features, y.vector, m_rho1, m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Clustered LR is unstable with C++11\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif
#else
	SG_WARNING("Please install Eigen3 to use MultitaskClusteredLogisticRegression\n")
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks);
	m_tasks_c = SGVector<float64_t>(options.n_tasks);
#endif

	SG_FREE(options.tasks_indices);

	return true;
}

}
