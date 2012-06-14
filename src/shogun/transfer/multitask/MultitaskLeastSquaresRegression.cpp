/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLeastSquaresRegression.h>
#include <shogun/lib/slep/slep_tree_mt_lsr.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CMultitaskLeastSquaresRegression::CMultitaskLeastSquaresRegression() :
	CSLEPMachine(), m_task_tree(NULL)
{
	register_parameters();
}

CMultitaskLeastSquaresRegression::CMultitaskLeastSquaresRegression(
     float64_t z, CDotFeatures* train_features, 
     CRegressionLabels* train_labels, CIndicesTree* task_tree) :
	CSLEPMachine(z,train_features,(CLabels*)train_labels), m_task_tree(NULL)
{
	set_task_tree(task_tree);
	register_parameters();
}

CMultitaskLeastSquaresRegression::~CMultitaskLeastSquaresRegression()
{
	SG_UNREF(m_task_tree);
}

void CMultitaskLeastSquaresRegression::register_parameters()
{
	SG_ADD((CSGObject**)&m_task_tree, "feature_tree", "feature tree", MS_NOT_AVAILABLE);
}

int32_t CMultitaskLeastSquaresRegression::get_current_task() const
{
	return m_current_task;
}

void CMultitaskLeastSquaresRegression::set_current_task(int32_t task)
{
	ASSERT(task>0);
	ASSERT(task<m_tasks_w.num_cols);
	m_current_task = task;
	int32_t n_feats = ((CDotFeatures*)features)->get_dim_feature_space();
	w = SGVector<float64_t>(n_feats);
	for (int32_t i=0; i<n_feats; i++)
		w[i] = m_tasks_w(i,task);
}

CIndicesTree* CMultitaskLeastSquaresRegression::get_task_tree() const
{
	SG_REF(m_task_tree);
	return m_task_tree;
}

void CMultitaskLeastSquaresRegression::set_task_tree(CIndicesTree* task_tree)
{
	SG_UNREF(m_task_tree);
	SG_REF(task_tree);
	m_task_tree = task_tree;
}

bool CMultitaskLeastSquaresRegression::train_machine(CFeatures* data)
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
	options.general = false;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;
	options.restart_num = 10000;
	options.n_nodes = m_task_tree->get_num_nodes();
	options.regularization = 0;
	SGVector<float64_t> ind = m_task_tree->get_ind();
	options.ind = ind.vector;
	options.G = NULL;
	options.initial_w = NULL;

	m_tasks_w = slep_tree_mt_lsr(features,y,m_z,options);

	SG_FREE(y);

	return true;
}

}
