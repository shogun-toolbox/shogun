/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/structure/DirectorStructuredModel.h>

#ifdef USE_SWIG_DIRECTORS

using namespace shogun;

CDirectorStructuredModel::CDirectorStructuredModel() : CStructuredModel()
{
}

CDirectorStructuredModel::~CDirectorStructuredModel()
{
}

int32_t CDirectorStructuredModel::get_dim() const
{
	SG_ERROR("Please implemement get_dim() in your target language before use\n");
	return 0;
}

CResultSet* CDirectorStructuredModel::argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training)
{
	SG_ERROR("Please implemement get_joint_feature_vector(feat_idx,lab_idx) in your target language before use\n");
	return NULL;
}

SGVector< float64_t > CDirectorStructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		int32_t lab_idx)
{
	SG_ERROR("Please implemement get_joint_feature_vector(feat_idx,lab_idx) in your target language before use\n");
	return SGVector<float64_t>();
}

SGVector< float64_t > CDirectorStructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	SG_ERROR("Please implemement get_joint_feature_vector(feat_idx,y) in your target language before use\n");
	return SGVector<float64_t>();
}

float64_t CDirectorStructuredModel::delta_loss(int32_t ytrue_idx, CStructuredData* ypred)
{
	SG_ERROR("Please implemement get_joint_feature_vector(feat_idx,y) in your target language before use\n");
	return 0.0;
}

float64_t CDirectorStructuredModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	SG_ERROR("Please implemement delta_loss(y1,y2) in your target language before use\n");
	return 0.0;
}
#endif /* USE_SWIG_DIRECTORS */
