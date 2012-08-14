/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/structure/MulticlassRiskFunction.h>

using namespace shogun;

CMulticlassRiskFunction::CMulticlassRiskFunction()
:CRiskFunction()
{
}

CMulticlassRiskFunction::~CMulticlassRiskFunction()
{
}

CMulticlassRiskData::CMulticlassRiskData()
:CRiskData()
{
}

CMulticlassRiskData::CMulticlassRiskData(
		CDotFeatures        *X,
		CMulticlassSOLabels *y,
		uint32_t            w_dim,
		uint32_t            nFeatures)
:CRiskData()
{
	m_X=X;
	m_y=y;
	m_w_dim=w_dim;
	m_nFeatures=nFeatures;
}

CMulticlassRiskData::~CMulticlassRiskData()
{
}

void CMulticlassRiskFunction::risk(
		void* 				data,
		float64_t* 			R,
		float64_t* 			subgrad,
		float64_t* 			W,
		TMultipleCPinfo* 	info)
{
	CMulticlassRiskData* data_struct=(CMulticlassRiskData*)data;
	CDotFeatures* X=data_struct->m_X;
	CMulticlassSOLabels* y=data_struct->m_y;
	uint32_t from, to;
	if (info)
	{
		from=info->from;
		to=(info->N == 0) ? X->get_num_vectors() : from+info->N;
	} else {
		from=0;
		to=X->get_num_vectors();
	}
	uint32_t num_classes=y->get_num_classes();
	uint32_t feats_dim=X->get_dim_feature_space();
	const uint32_t w_dim=data_struct->m_w_dim;

	*R=0;
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
		*R += Rmax;

		X->add_to_dense_vec(1.0, i, subgrad+yhat*feats_dim, feats_dim);
		X->add_to_dense_vec(-1.0, i, subgrad+GT*feats_dim, feats_dim);
	}
}
