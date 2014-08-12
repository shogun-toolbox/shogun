/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <shogun/structure/SOSVMHelper.h>
#include <shogun/base/Parameter.h>
#include <shogun/labels/StructuredLabels.h>

using namespace shogun;

CSOSVMHelper::CSOSVMHelper() : CSGObject()
{
	init();
}

CSOSVMHelper::CSOSVMHelper(int32_t bufsize) : CSGObject()
{
	m_bufsize = bufsize;
	init();
}

CSOSVMHelper::~CSOSVMHelper()
{
}

void CSOSVMHelper::init()
{
	SG_ADD(&m_primal, "primal", "History of primal values", MS_NOT_AVAILABLE);
	SG_ADD(&m_dual, "dual", "History of dual values", MS_NOT_AVAILABLE);
	SG_ADD(&m_duality_gap, "duality_gap", "History of duality gaps", MS_NOT_AVAILABLE);
	SG_ADD(&m_eff_pass, "eff_pass", "Effective passes", MS_NOT_AVAILABLE);
	SG_ADD(&m_train_error, "train_error", "History of training errors", MS_NOT_AVAILABLE);
	SG_ADD(&m_tracker, "tracker", "Tracker of training progress", MS_NOT_AVAILABLE);
	SG_ADD(&m_bufsize, "bufsize", "Buffer size", MS_NOT_AVAILABLE);

	m_tracker = 0;
	m_bufsize = 1000;
	m_primal = SGVector<float64_t>(m_bufsize);
	m_dual = SGVector<float64_t>(m_bufsize);
	m_duality_gap = SGVector<float64_t>(m_bufsize);
	m_eff_pass = SGVector<float64_t>(m_bufsize);
	m_train_error = SGVector<float64_t>(m_bufsize);
	m_primal.zero();
	m_dual.zero();
	m_duality_gap.zero();
	m_eff_pass.zero();
	m_train_error.zero();
}

float64_t CSOSVMHelper::primal_objective(SGVector<float64_t> w, CStructuredModel* model, float64_t lbda)
{
	float64_t hinge_losses = 0.0;
	CStructuredLabels* labels = model->get_labels();
	int32_t N = labels->get_num_labels();
	SG_UNREF(labels);

	for (int32_t i = 0; i < N; i++)
	{
		// solve the loss-augmented inference for point i
		CResultSet* result = model->argmax(w, i);

		// hinge loss for point i
		float64_t hinge_loss_i = result->score;
		
		if (hinge_loss_i < 0)
			hinge_loss_i = 0;

		hinge_losses += hinge_loss_i;

		SG_UNREF(result);
	}

	return (lbda/2 * SGVector<float64_t>::dot(w.vector, w.vector, w.vlen) + hinge_losses/N);
}

float64_t CSOSVMHelper::dual_objective(SGVector<float64_t> w, float64_t aloss, float64_t lbda)
{
	return (-lbda/2 * SGVector<float64_t>::dot(w.vector, w.vector, w.vlen) + aloss);
}

float64_t CSOSVMHelper::average_loss(SGVector<float64_t> w, CStructuredModel* model, bool is_ub)
{
	float64_t loss = 0.0;
	CStructuredLabels* labels = model->get_labels();
	int32_t N = labels->get_num_labels();
	SG_UNREF(labels);

	for (int32_t i = 0; i < N; i++)
	{
		// solve the standard inference for point i
		CResultSet* result = model->argmax(w, i, is_ub);

		loss += result->delta;

		SG_UNREF(result);
	}

	return loss / N;
}

void CSOSVMHelper::add_debug_info(float64_t primal, float64_t eff_pass, float64_t train_error,
		float64_t dual, float64_t dgap)
{
	if (m_tracker >= m_bufsize)
	{
		SG_PRINT("%s::add_debug_information(): Buffer overflows! No more values will be recorded!\n",
			get_name());

		return;
	}

	m_primal[m_tracker] = primal;
	m_eff_pass[m_tracker] = eff_pass;
	m_train_error[m_tracker] = train_error;

	if (dgap >= 0)
	{
		m_dual[m_tracker] = dual;
		m_duality_gap[m_tracker] = dgap;
	}

	m_tracker++;
}

SGVector<float64_t> CSOSVMHelper::get_primal_values() const
{
	return m_primal;
}

SGVector<float64_t> CSOSVMHelper::get_dual_values() const
{
	return m_dual;
}

SGVector<float64_t> CSOSVMHelper::get_duality_gaps() const
{
	return m_duality_gap;
}

SGVector<float64_t> CSOSVMHelper::get_eff_passes() const
{
	return m_eff_pass;
}

SGVector<float64_t> CSOSVMHelper::get_train_errors() const
{
	return m_train_error;
}

void CSOSVMHelper::terminate()
{
	m_primal.resize_vector(m_tracker);
	m_dual.resize_vector(m_tracker);
	m_duality_gap.resize_vector(m_tracker);
	m_eff_pass.resize_vector(m_tracker);
	m_train_error.resize_vector(m_tracker);
}
