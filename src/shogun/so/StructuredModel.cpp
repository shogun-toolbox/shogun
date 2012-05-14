/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/so/StructuredModel.h>

using namespace shogun;

CStructuredModel::CStructuredModel()
{
}

CStructuredModel::~CStructuredModel()
{
}

/* TODO */
void CStructuredModel::init()
{
}

/* TODO */
int32_t CStructuredModel::get_dim()
{
}

SGVector< float64_t > CStructuredModel::compute_combined_feature(int32_t feat_idx, int32_t lab_idx)
{
	return m_compute_combined_feature(m_features, m_labels, feat_idx, lab_idx);
}

CResultSet* CStructuredModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	return m_argmax(m_features, m_labels, w, feat_idx);
}


float64_t CStructuredModel::compute_delta_loss(CStructuredLabels* labels, CStructuredData ypred, int32_t ytrue_id)
{
	return m_compute_delta_loss(labels, ypred, ytrue_id);
}
