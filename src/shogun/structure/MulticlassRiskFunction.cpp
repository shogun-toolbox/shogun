/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/structure/MulticlassRiskFunction.h>
#include <shogun/structure/libbmrm.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/structure/MulticlassModel.h>

using namespace shogun;

CMulticlassRiskFunction::CMulticlassRiskFunction()
:CRiskFunction()
{
}

CMulticlassRiskFunction::~CMulticlassRiskFunction()
{
}

void CMulticlassRiskFunction::risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W)
{
	bmrm_data_T* data_struct=(bmrm_data_T*)data;
	CDotFeatures* X=(CDotFeatures*)data_struct->X;
	CMulticlassSOLabels* y=(CMulticlassSOLabels*)data_struct->y;

	uint32_t N=X->get_num_vectors();
	uint32_t num_classes=y->get_num_classes();
	uint32_t feats_dim=X->get_dim_feature_space();
	uint32_t w_dim=(uint32_t)data_struct->w_dim;

	*R=0;
	SGVector< float64_t > subgradient(w_dim);
	subgradient.zero();

	SGVector< float64_t > xi;

	float64_t Rtmp=0.0;
	float64_t Rmax=0.0;
	float64_t loss=0.0;
	uint32_t yhat=0;
	uint32_t GT=0;

	/* loop through examples */
	for(uint32_t i = 0; i < N; ++i)
	{
		Rmax=-CMath::INFTY;
		xi=X->get_computed_dot_feature_vector(i);
		GT=(uint32_t)((CRealNumber*)y->get_label(i))->value;

		for (uint32_t c = 0; c < num_classes; ++c)
		{
			loss=(c == GT) ? 0.0 : 1.0;
			Rtmp = loss + SGVector< float64_t >::dot(W+c*feats_dim, xi.vector, feats_dim) - SGVector< float64_t >::dot(W+GT*feats_dim, xi.vector, feats_dim);

			if (Rtmp > Rmax)
			{
				Rmax=Rtmp;
				yhat=c;
			}
		}
		*R += Rmax;

		for(uint32_t j = 0; j < feats_dim; ++j)
		{
			subgradient[yhat*feats_dim+j]+=xi[j];
			subgradient[GT*feats_dim+j]-=xi[j];
		}

	}
	memcpy(subgrad, subgradient.vector, w_dim*sizeof(float64_t));
}
