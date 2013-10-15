/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/MultitaskLeastSquaresRegression.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/lib/slep/slep_solver.h>
#include <shogun/lib/slep/slep_options.h>

namespace shogun
{

CMultitaskLeastSquaresRegression::CMultitaskLeastSquaresRegression() :
	CMultitaskLinearMachine()
{
	initialize_parameters();
	register_parameters();
}

CMultitaskLeastSquaresRegression::CMultitaskLeastSquaresRegression(
     float64_t z, CDotFeatures* train_features,
     CRegressionLabels* train_labels, CTaskRelation* task_relation) :
	CMultitaskLinearMachine(train_features,(CLabels*)train_labels,task_relation)
{
	set_z(z);
	initialize_parameters();
	register_parameters();
}

CMultitaskLeastSquaresRegression::~CMultitaskLeastSquaresRegression()
{
}

void CMultitaskLeastSquaresRegression::register_parameters()
{
	SG_ADD(&m_z, "z", "regularization coefficient", MS_AVAILABLE);
	SG_ADD(&m_q, "q", "q of L1/Lq", MS_AVAILABLE);
	SG_ADD(&m_termination, "termination", "termination", MS_NOT_AVAILABLE);
	SG_ADD(&m_regularization, "regularization", "regularization", MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "tolerance", "tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iter, "max_iter", "maximum number of iterations", MS_NOT_AVAILABLE);
}

void CMultitaskLeastSquaresRegression::initialize_parameters()
{
	set_z(0.0);
	set_q(2.0);
	set_termination(0);
	set_regularization(0);
	set_tolerance(1e-3);
	set_max_iter(1000);
}

bool CMultitaskLeastSquaresRegression::train_locked_implementation(SGVector<index_t>* tasks)
{
	SG_NOTIMPLEMENTED
	return false;
}

float64_t CMultitaskLeastSquaresRegression::apply_one(int32_t i)
{
	float64_t dot = features->dense_dot(i,m_tasks_w.get_column_vector(m_current_task),m_tasks_w.num_rows);
	return dot + m_tasks_c[m_current_task];
}

int32_t CMultitaskLeastSquaresRegression::get_max_iter() const
{
	return m_max_iter;
}
int32_t CMultitaskLeastSquaresRegression::get_regularization() const
{
	return m_regularization;
}
int32_t CMultitaskLeastSquaresRegression::get_termination() const
{
	return m_termination;
}
float64_t CMultitaskLeastSquaresRegression::get_tolerance() const
{
	return m_tolerance;
}
float64_t CMultitaskLeastSquaresRegression::get_z() const
{
	return m_z;
}
float64_t CMultitaskLeastSquaresRegression::get_q() const
{
	return m_q;
}

void CMultitaskLeastSquaresRegression::set_max_iter(int32_t max_iter)
{
	ASSERT(max_iter>=0)
	m_max_iter = max_iter;
}
void CMultitaskLeastSquaresRegression::set_regularization(int32_t regularization)
{
	ASSERT(regularization==0 || regularization==1)
	m_regularization = regularization;
}
void CMultitaskLeastSquaresRegression::set_termination(int32_t termination)
{
	ASSERT(termination>=0 && termination<=4)
	m_termination = termination;
}
void CMultitaskLeastSquaresRegression::set_tolerance(float64_t tolerance)
{
	ASSERT(tolerance>0.0)
	m_tolerance = tolerance;
}
void CMultitaskLeastSquaresRegression::set_z(float64_t z)
{
	m_z = z;
}
void CMultitaskLeastSquaresRegression::set_q(float64_t q)
{
	m_q = q;
}

bool CMultitaskLeastSquaresRegression::train_machine(CFeatures* data)
{
	if (data && (CDotFeatures*)data)
		set_features((CDotFeatures*)data);

	ASSERT(features)
	ASSERT(m_labels)

	SGVector<float64_t> y = ((CRegressionLabels*)m_labels)->get_labels();

	slep_options options = slep_options::default_options();
	options.n_tasks = m_task_relation->get_num_tasks();
	options.tasks_indices = m_task_relation->get_tasks_indices();
	options.q = m_q;
	options.regularization = m_regularization;
	options.termination = m_termination;
	options.tolerance = m_tolerance;
	options.max_iter = m_max_iter;

	ETaskRelationType relation_type = m_task_relation->get_relation_type();
	switch (relation_type)
	{
		case TASK_GROUP:
		{
			//CTaskGroup* task_group = (CTaskGroup*)m_task_relation;
			options.mode = MULTITASK_GROUP;
			options.loss = LEAST_SQUARES;
			m_tasks_w = slep_solver(features, y.vector, m_z, options).w;
			m_tasks_c = SGVector<float64_t>(options.n_tasks);
			m_tasks_c.zero();
		}
		break;
		case TASK_TREE:
		{
			CTaskTree* task_tree = (CTaskTree*)m_task_relation;
			SGVector<float64_t> ind_t = task_tree->get_SLEP_ind_t();
			options.ind_t = ind_t.vector;
			options.n_nodes = ind_t.vlen/3;
			options.mode = MULTITASK_TREE;
			options.loss = LEAST_SQUARES;
			m_tasks_w = slep_solver(features, y.vector, m_z, options).w;
			m_tasks_c = SGVector<float64_t>(options.n_tasks);
			m_tasks_c.zero();
		}
		break;
		default:
			SG_ERROR("Not supported task relation type\n")
	}

	SG_FREE(options.tasks_indices);

	return true;
}

}
