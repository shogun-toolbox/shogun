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
#include <shogun/structure/HMSVMLabels.h>

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
SGVector< float64_t > CHMSVMModel::get_joint_feature_vector(
		int32_t feat_idx,
		CStructuredData* y)
{
	return SGVector< float64_t >();
}

/* TODO */
CResultSet* CHMSVMModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	return NULL;
}

float64_t CHMSVMModel::delta_loss(int32_t ytrue_idx, CStructuredData* ypred)
{
	if ( ytrue_idx < 0 || ytrue_idx >= m_labels->get_num_labels() )
		SG_ERROR("The label index must be inside [0, num_labels-1]\n");

	if ( ypred->get_structured_data_type() != SDT_SEQUENCE )
		SG_ERROR("ypred must be a CSequence\n");

	// Shorthand for ypred with the correct structured data type
	CSequence* yhat  = CSequence::obtain_from_generic(ypred);
	// The same for ytrue
	CSequence* ytrue = CSequence::obtain_from_generic(
			m_labels->get_label(ytrue_idx));

	// Compute the Hamming loss, number of distinct elements in the sequences
	ASSERT( yhat->data.vlen == ytrue->data.vlen );

	float64_t out = 0.0;
	for ( int32_t i = 0 ; i < yhat->data.vlen ; ++i )
		out += yhat->data[i] != ytrue->data[i];

	return out;
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
