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
	init();
}

	CMulticlassModel::CMulticlassModel(CFeatures* features, CStructuredLabels* labels)
: CStructuredModel(features, labels)
{
	init();
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
	CRealNumber* r = CRealNumber::obtain_from_generic(y);
	ASSERT(r != NULL)
	float64_t label_value = r->value;

	for ( index_t i = 0, j = label_value*x.vlen ; i < x.vlen ; ++i, ++j )
		psi[j] = x[i];

	return psi;
}

CResultSet* CMulticlassModel::argmax(
		SGVector< float64_t > w,
		int32_t feat_idx,
		bool const training)
{
	CDotFeatures* df = (CDotFeatures*) m_features;
	int32_t feats_dim   = df->get_dim_feature_space();

	if ( training )
	{
		CMulticlassSOLabels* ml = (CMulticlassSOLabels*) m_labels;
		m_num_classes = ml->get_num_classes();
	}
	else
	{
		REQUIRE(m_num_classes > 0, "The model needs to be trained before "
				"using it for prediction\n");
	}

	ASSERT(feats_dim*m_num_classes == w.vlen);

	// Find the class that gives the maximum score

	float64_t score = 0, ypred = 0;
	float64_t max_score = -CMath::INFTY;

	for ( int32_t c = 0 ; c < m_num_classes ; ++c )
	{
		score = df->dense_dot(feat_idx, w.vector+c*feats_dim, feats_dim);
		if ( training )
			score += delta_loss(feat_idx, c);

		if ( score > max_score )
		{
			max_score = score;
			ypred = c;
		}
	}

	// Build the CResultSet object to return
	CResultSet* ret = new CResultSet();
	CRealNumber* y  = new CRealNumber(ypred);
	SG_REF(ret);
	SG_REF(y);

	ret->psi_pred = get_joint_feature_vector(feat_idx, y);
	ret->score    = max_score;
	ret->argmax   = y;
	if ( training )
	{
		ret->psi_truth = CStructuredModel::get_joint_feature_vector(feat_idx, feat_idx);
		ret->delta     = CStructuredModel::delta_loss(feat_idx, y);
	}

	return ret;
}

float64_t CMulticlassModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	CRealNumber* rn1 = CRealNumber::obtain_from_generic(y1);
	CRealNumber* rn2 = CRealNumber::obtain_from_generic(y2);
	ASSERT(rn1 != NULL);
	ASSERT(rn2 != NULL);

	return delta_loss(rn1->value, rn2->value);
}

float64_t CMulticlassModel::delta_loss(int32_t y1_idx, float64_t y2)
{
	REQUIRE(y1_idx >= 0 || y1_idx < m_labels->get_num_labels(),
			"The label index must be inside [0, num_labels-1]\n");

	CRealNumber* rn1 = CRealNumber::obtain_from_generic(m_labels->get_label(y1_idx));
	float64_t ret = delta_loss(rn1->value, y2);
	SG_UNREF(rn1);

	return ret;
}

float64_t CMulticlassModel::delta_loss(float64_t y1, float64_t y2)
{
	return (y1 == y2) ? 0 : 1;
}

void CMulticlassModel::init_opt(
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
	C = SGMatrix< float64_t >::create_identity_matrix(get_dim(), 1);
}

void CMulticlassModel::init()
{
	SG_ADD(&m_num_classes, "m_num_classes", "The number of classes",
			MS_NOT_AVAILABLE);

	m_num_classes = 0;
}

float64_t CMulticlassModel::risk(float64_t* subgrad, float64_t* W, TMultipleCPinfo* info)
{
	CDotFeatures* X=(CDotFeatures*)m_features;
	CMulticlassSOLabels* y=(CMulticlassSOLabels*)m_labels;
	m_num_classes = y->get_num_classes();
	uint32_t from, to;

	if (info)
	{
		from=info->_from;
		to=(info->N == 0) ? X->get_num_vectors() : from+info->N;
	} else {
		from=0;
		to=X->get_num_vectors();
	}

	uint32_t num_classes=y->get_num_classes();
	uint32_t feats_dim=X->get_dim_feature_space();
	const uint32_t w_dim=get_dim();

	float64_t R=0.0;
	for (uint32_t i=0; i<w_dim; i++)
		subgrad[i] = 0;

	float64_t Rtmp=0.0;
	float64_t Rmax=0.0;
	float64_t loss=0.0;
	uint32_t yhat=0;
	uint32_t GT=0;

	/* loop through examples */
	for(uint32_t i=from; i<to; ++i)
	{
		Rmax=-CMath::INFTY;
		GT=(uint32_t)((CRealNumber*)y->get_label(i))->value;

		for (uint32_t c = 0; c < num_classes; ++c)
		{
			loss=(c == GT) ? 0.0 : 1.0;
			Rtmp=loss+X->dense_dot(i, W+c*feats_dim, feats_dim)
				-X->dense_dot(i, W+GT*feats_dim, feats_dim);

			if (Rtmp > Rmax)
			{
				Rmax=Rtmp;
				yhat=c;
			}
		}
		R += Rmax;

		X->add_to_dense_vec(1.0, i, subgrad+yhat*feats_dim, feats_dim);
		X->add_to_dense_vec(-1.0, i, subgrad+GT*feats_dim, feats_dim);
	}

	return R;
}

