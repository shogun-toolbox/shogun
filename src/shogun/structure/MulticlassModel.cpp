/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

CMulticlassModel::CMulticlassModel()
: CStructuredModel()
{
}

CMulticlassModel::CMulticlassModel(CFeatures* features, CStructuredLabels* labels)
: CStructuredModel(features, labels)
{
}

CMulticlassModel::~CMulticlassModel()
{
}

int32_t CMulticlassModel::get_dim() const
{
	// TODO make the casts safe!
	int32_t num_classes = ((CMulticlassSOLabels*) m_labels)->get_num_classes();
	int32_t feats_dim   = ((CDotFeatures*) m_features)->get_dim_feature_space();

	return feats_dim*num_classes;
}

SGVector< float64_t > CMulticlassModel::get_joint_feature_vector(int32_t feat_idx, CStructuredData* y)
{
	SGVector< float64_t > psi( get_dim() );
	psi.zero();

	SGVector< float64_t > x = ((CDotFeatures*) m_features)->
				get_computed_dot_feature_vector(feat_idx);
	/* TODO add checks for the casting!! */
	float64_t label_value = CRealNumber::obtain_from_generic(y)->value;

	for ( index_t i = 0, j = label_value*x.vlen ; i < x.vlen ; ++i, ++j )
		psi[j] = x[i];

	return psi;
}

CResultSet* CMulticlassModel::argmax(SGVector< float64_t > w, int32_t feat_idx)
{
	CDotFeatures* df = (CDotFeatures*) m_features;
	CMulticlassSOLabels* ml = (CMulticlassSOLabels*) m_labels;

	int32_t feats_dim   = df->get_dim_feature_space();
	int32_t num_classes = ml->get_num_classes();

	ASSERT(feats_dim*num_classes == w.vlen);

	// Find the class that gives the maximum score

	float64_t score = 0, ypred = 0;
	float64_t max_score = df->dense_dot(feat_idx, w.vector, feats_dim);
	
	for ( int32_t c = 1 ; c < num_classes ; ++c )
	{
		score = df->dense_dot(feat_idx, w.vector+c*feats_dim, feats_dim);

		if ( score > max_score )
		{
			score = max_score;
			ypred = c;
		}
	}

	// Build the CResultSet object to return
	CResultSet* ret = new CResultSet();
	CRealNumber* y  = new CRealNumber(ypred);
	SG_REF(ret);
	SG_REF(y);

	ret->psi_truth = CStructuredModel::get_joint_feature_vector(feat_idx, feat_idx);
	ret->psi_pred  = get_joint_feature_vector(feat_idx, y);
	ret->score     = max_score;
	ret->delta     = CStructuredModel::delta_loss(feat_idx, y);
	ret->argmax    = y;

	return ret;
}

float64_t CMulticlassModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	CRealNumber* rn1 = CRealNumber::obtain_from_generic(y1);
	CRealNumber* rn2 = CRealNumber::obtain_from_generic(y2);

	return ( rn1->value == rn2->value ) ? 0 : 1;
}

void CMulticlassModel::init_opt(
		SGMatrix< float64_t > A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
	C = SGMatrix< float64_t >::create_identity_matrix(get_dim(), 1);
}
