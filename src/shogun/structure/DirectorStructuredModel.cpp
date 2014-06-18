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
	SG_ERROR("Please implemement get_dim() in your target language before use\n")
	return 0;
}

CResultSet* CDirectorStructuredModel::argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training)
{
	SG_ERROR("Please implemement argmax(w,feat_idx,lab_idx,training) in your target language before use\n")
	return NULL;
}

SGVector< float64_t > CDirectorStructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	SG_ERROR("Please implemement get_joint_feature_vector(feat_idx,y) in your target language before use\n")
	return SGVector<float64_t>();
}

float64_t CDirectorStructuredModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	SG_ERROR("Please implemement delta_loss(y1,y2) in your target language before use\n")
	return 0.0;
}

bool CDirectorStructuredModel::check_training_setup() const
{
	SG_ERROR("Please implemement check_trainig_setup() in your target language before use\n")
	return false;
}

void CDirectorStructuredModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
	SG_ERROR("Please implemement init_primal_opt(regularization,A,a,B,b,lb,ub,C) in your target language before use\n")
}

void CDirectorStructuredModel::init_training()
{
	SG_ERROR("Please implemement init_training() in your target language before use\n")
}

#endif /* USE_SWIG_DIRECTORS */
