/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/HMSVMModel.h>

using namespace shogun;

CHMSVMModel::CHMSVMModel()
: CStructuredModel()
{
}

CHMSVMModel::CHMSVMModel(CFeatures* features, CStructuredLabels* labels)
: CStructuredModel(features, labels)
{
}

CHMSVMModel::~CHMSVMModel()
{
}

/* TODO */
int32_t CHMSVMModel::get_dim() const
{
	return 0;
}

/* TODO */
SGVector< float64_t > CHMSVMModel::get_joint_feature_vector(int32_t feat_idx, CStructuredData* y)
{
	return SGVector< float64_t >();
}

/* TODO */
CResultSet* CHMSVMModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	return NULL;
}

/* TODO */
float64_t CHMSVMModel::delta_loss(int32_t ytrue_idx, CStructuredData* ypred)
{
	return 0.0;
}

/* TODO */
void CHMSVMModel::init_opt(
		SGMatrix< float64_t > A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
}
