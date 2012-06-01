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

CStructuredModel::CStructuredModel() : CSGObject()
{
}

CStructuredModel::~CStructuredModel()
{
	SG_UNREF(m_labels);
	SG_UNREF(m_features);
}

/* TODO */
void CStructuredModel::init_opt(
		SGMatrix< float64_t > A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > C)
{
}


/* TODO */
int32_t CStructuredModel::get_dim()
{
	return 0;
}

void CStructuredModel::set_labels(CStructuredLabels* labs)
{
	SG_UNREF(m_labels);
	SG_REF(labs);
	m_labels = labs;
}

void CStructuredModel::set_features(CFeatures* feats)
{
	SG_UNREF(m_features);
	SG_REF(feats);
	m_features = feats;
}

/** TODO */
SGVector< float64_t > CStructuredModel::compute_joint_feature(
		int32_t feat_idx, 
		int32_t lab_idx)
{
	SG_ERROR("CStructuredModel::compute_combined_feature not implemented "
		 "yet");
	return SGVector< float64_t >();
}

CResultSet* CStructuredModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	return m_argmax->argmax(m_features, feat_idx, m_labels, w);
}


float64_t CStructuredModel::compute_delta_loss(
		CStructuredLabels* labels, 
		int32_t ytrue_id, 
		CStructuredData ypred)
{
	return m_loss->loss(labels, ytrue_id, ypred);
}

void CStructuredModel::init()
{
	SG_ADD((CSGObject**) &m_labels, "m_labels", "Structured labels", 
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_features, "m_features", "Feature vectors", 
			MS_NOT_AVAILABLE);
	//TODO add rest of members when function pointers removed

	m_features = NULL;
	m_labels   = NULL;
}
