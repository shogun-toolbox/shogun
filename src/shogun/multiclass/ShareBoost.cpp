/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <algorithm>

#include <lib/Time.h>
#include <mathematics/Math.h>
#include <multiclass/ShareBoost.h>
#include <multiclass/ShareBoostOptimizer.h>
#include <features/DenseSubsetFeatures.h>
#include <labels/RegressionLabels.h>

using namespace shogun;

CShareBoost::CShareBoost()
	:CLinearMulticlassMachine(), m_nonzero_feas(0)
{
	init_sb_params();
}

CShareBoost::CShareBoost(CDenseFeatures<float64_t> *features, CMulticlassLabels *labs, int32_t num_nonzero_feas)
	:CLinearMulticlassMachine(new CMulticlassOneVsRestStrategy(), features, NULL, labs), m_nonzero_feas(num_nonzero_feas)
{
	init_sb_params();
}

void CShareBoost::init_sb_params()
{
	SG_ADD(&m_nonzero_feas, "m_nonzero_feas", "Number of non-zero features", MS_NOT_AVAILABLE);
}

SGVector<int32_t> CShareBoost::get_activeset()
{
	return m_activeset.clone();
}

bool CShareBoost::train_machine(CFeatures* data)
{
	if (data)
		set_features(data);

	if (m_features == NULL)
		SG_ERROR("No features given for training\n")
	if (m_labels == NULL)
		SG_ERROR("No labels given for training\n")

	if (m_nonzero_feas <= 0)
		SG_ERROR("Set a valid (> 0) number of non-zero features to seek before training\n")
	if (m_nonzero_feas >= dynamic_cast<CDenseFeatures<float64_t>*>(m_features)->get_num_features())
		SG_ERROR("It doesn't make sense to use ShareBoost with num non-zero features >= num features in the data\n")

	m_fea = dynamic_cast<CDenseFeatures<float64_t> *>(m_features)->get_feature_matrix();
	m_rho = SGMatrix<float64_t>(m_multiclass_strategy->get_num_classes(), m_fea.num_cols);
	m_rho_norm = SGVector<float64_t>(m_fea.num_cols);
	m_pred = SGMatrix<float64_t>(m_fea.num_cols, m_multiclass_strategy->get_num_classes());
	m_pred.zero();

	m_activeset = SGVector<int32_t>(m_fea.num_rows);
	m_activeset.vlen = 0;

	m_machines->reset_array();
	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
		m_machines->push_back(new CLinearMachine());

	CTime *timer = new CTime();

	float64_t t_compute_pred = 0; // t of 1st round is 0, since no pred to compute
	for (int32_t t=0; t < m_nonzero_feas; ++t)
	{
		timer->start();
		compute_rho();
		int32_t i_fea = choose_feature();
		m_activeset.vector[m_activeset.vlen] = i_fea;
		m_activeset.vlen += 1;
		float64_t t_choose_feature = timer->cur_time_diff();
		timer->start();
		optimize_coefficients();
		float64_t t_optimize = timer->cur_time_diff();

		SG_SDEBUG(" SB[round %03d]: (%8.4f + %8.4f) sec.\n", t,
				t_compute_pred + t_choose_feature, t_optimize);

		timer->start();
		compute_pred();
		t_compute_pred = timer->cur_time_diff();
	}

	SG_UNREF(timer);

	// release memory
	m_fea = SGMatrix<float64_t>();
	m_rho = SGMatrix<float64_t>();
	m_rho_norm = SGVector<float64_t>();
	m_pred = SGMatrix<float64_t>();

	return true;
}

void CShareBoost::compute_pred()
{
	CDenseFeatures<float64_t> *fea = dynamic_cast<CDenseFeatures<float64_t> *>(m_features);
	CDenseSubsetFeatures<float64_t> *subset_fea = new CDenseSubsetFeatures<float64_t>(fea, m_activeset);
	SG_REF(subset_fea);
	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
	{
		CLinearMachine *machine = dynamic_cast<CLinearMachine *>(m_machines->get_element(i));
		CRegressionLabels *lab = machine->apply_regression(subset_fea);
		SGVector<float64_t> lab_raw = lab->get_labels();
		std::copy(lab_raw.vector, lab_raw.vector + lab_raw.vlen, m_pred.get_column_vector(i));
		SG_UNREF(machine);
		SG_UNREF(lab);
	}
	SG_UNREF(subset_fea);
}

void CShareBoost::compute_pred(const float64_t *W)
{
	int32_t w_len = m_activeset.vlen;

	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
	{
		CLinearMachine *machine = dynamic_cast<CLinearMachine *>(m_machines->get_element(i));
		SGVector<float64_t> w(w_len);
		std::copy(W + i*w_len, W + (i+1)*w_len, w.vector);
		machine->set_w(w);
		SG_UNREF(machine);
	}
	compute_pred();
}

void CShareBoost::compute_rho()
{
	CMulticlassLabels *lab = dynamic_cast<CMulticlassLabels *>(m_labels);
	for (int32_t i=0; i < m_rho.num_rows; ++i)
	{ // i loop classes
		for (int32_t j=0; j < m_rho.num_cols; ++j)
		{ // j loop samples
			int32_t label = lab->get_int_label(j);

			m_rho(i,j) = CMath::exp((label == i) - m_pred(j, label) + m_pred(j, i));
		}
	}

	// normalize
	for (int32_t j=0; j < m_rho.num_cols; ++j)
	{
		m_rho_norm[j] = 0;
		for (int32_t i=0; i < m_rho.num_rows; ++i)
			m_rho_norm[j] += m_rho(i,j);
	}
}

int32_t CShareBoost::choose_feature()
{
	SGVector<float64_t> l1norm(m_fea.num_rows);
	for (int32_t j=0; j < m_fea.num_rows; ++j)
	{
		if (std::find(&m_activeset[0], &m_activeset[m_activeset.vlen], j) !=
				&m_activeset[m_activeset.vlen])
		{
			l1norm[j] = 0;
		}
		else
		{
			l1norm[j] = 0;
			CMulticlassLabels *lab = dynamic_cast<CMulticlassLabels *>(m_labels);
			for (int32_t k=0; k < m_multiclass_strategy->get_num_classes(); ++k)
			{
				float64_t abssum = 0;
				for (int32_t ii=0; ii < m_fea.num_cols; ++ii)
				{
					abssum += m_fea(j, ii)*(m_rho(k, ii)/m_rho_norm[ii] -
							(j == lab->get_int_label(ii)));
				}
				l1norm[j] += CMath::abs(abssum);
			}
			l1norm[j] /= m_fea.num_cols;
		}
	}

	return SGVector<float64_t>::arg_max(l1norm.vector, 1, l1norm.vlen);
}

void CShareBoost::optimize_coefficients()
{
	ShareBoostOptimizer optimizer(this, false);
	optimizer.optimize();
}

void CShareBoost::set_features(CFeatures *f)
{
	CDenseFeatures<float64_t> *fea = dynamic_cast<CDenseFeatures<float64_t> *>(f);
	if (fea == NULL)
		SG_ERROR("Require DenseFeatures<float64_t>\n")
	CLinearMulticlassMachine::set_features(fea);
}
