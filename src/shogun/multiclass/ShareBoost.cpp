/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/ShareBoost.h>

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

bool CShareBoost::train_machine(CFeatures* data)
{
	if (data)
		set_features(data);

	if (m_features == NULL)
		SG_ERROR("No features given for training\n");
	if (m_labels == NULL)
		SG_ERROR("No labels given for training\n");

	if (m_nonzero_feas <= 0)
		SG_ERROR("Set a valid (> 0) number of non-zero features to seek before training\n");
	if (m_nonzero_feas >= dynamic_cast<CDenseFeatures<float64_t>*>(m_features)->get_num_features())
		SG_ERROR("It doesn't make sense to use ShareBoost with num non-zero features >= num features in the data\n");

	m_fea = dynamic_cast<CDenseFeatures<float64_t> *>(m_features)->get_feature_matrix();
	m_rho = SGMatrix<float64_t>(m_multiclass_strategy->get_num_classes(), m_fea.num_cols);
	m_pred = SGMatrix<float64_t>(m_fea.num_cols, m_multiclass_strategy->get_num_classes());

	m_activeset = SGVector<int32_t>(m_fea.num_rows);
	m_activeset.vlen = 0;

	m_machines->reset_array();
	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
		m_machines->push_back(new CLinearMachine());

	for (int32_t t=0; t < m_nonzero_feas; ++t)
	{
		compute_rho();
		int32_t i_fea = choose_feature();
		m_activeset.vector[m_activeset.vlen] = i_fea;
		m_activeset.vlen += 1;
		optimize_coefficients();
		compute_pred();
	}

	// release memory
	m_fea = SGMatrix<float64_t>();
	m_rho = SGMatrix<float64_t>();

	return true;
}

void CShareBoost::compute_pred()
{

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
		float64_t sum = 0;
		for (int32_t i=0; i < m_rho.num_rows; ++i)
			sum += m_rho(i,j);

		for (int32_t i=0; i < m_rho.num_rows; ++i)
			m_rho(i,j) /= sum;
	}
}

int32_t CShareBoost::choose_feature()
{
	return 0;
}

void CShareBoost::optimize_coefficients()
{
}

void CShareBoost::set_features(CFeatures *f)
{
	CDenseFeatures<float64_t> *fea = dynamic_cast<CDenseFeatures<float64_t> *>(f);
	if (fea == NULL)
		SG_ERROR("Require DenseFeatures<float64_t>\n");
	CLinearMulticlassMachine::set_features(fea);
}
