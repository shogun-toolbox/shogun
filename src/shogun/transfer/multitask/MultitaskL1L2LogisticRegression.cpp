/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskL1L2LogisticRegression.h>
#include <shogun/lib/slep/malsar_joint_feature_learning.h>
#include <shogun/lib/slep/slep_options.h>
#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

CMultitaskL1L2LogisticRegression::CMultitaskL1L2LogisticRegression() :
	CMultitaskLogisticRegression(), m_rho1(0.0), m_rho2(0.0)
{
}

CMultitaskL1L2LogisticRegression::CMultitaskL1L2LogisticRegression(
     float64_t rho1, float64_t rho2, CDotFeatures* train_features, 
     CBinaryLabels* train_labels, CIndexBlockRelation* task_relation) :
	CMultitaskLogisticRegression(0.0,train_features,train_labels,task_relation)
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
	slep_result_t model = malsar_joint_feature_learning(
		features, y.vector, m_rho1, m_rho2, options);

	m_tasks_w = model.w;
	m_tasks_c = model.c;
#else
	SG_WARNING("Please install Eigen3 to use MultitaskL1L2LogisticRegression\n");
	m_tasks_w = SGMatrix<float64_t>(((CDotFeatures*)features)->get_dim_feature_space(), options.n_tasks); 
	m_tasks_c = SGVector<float64_t>(options.n_tasks); 
#endif
	
	ASSERT(m_task_relation);

	return true;
}

}
