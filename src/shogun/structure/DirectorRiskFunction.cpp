/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/structure/DirectorRiskFunction.h>
#include <shogun/structure/libbmrm.h>

using namespace shogun;

CDirectorRiskFunction::CDirectorRiskFunction()
: CRiskFunction()
{
}

CDirectorRiskFunction::~CDirectorRiskFunction()
{
}

void CDirectorRiskFunction::risk(void* data, float64_t* R, float64_t* subgrad, float64_t* W)
{
	CDotFeatures* features = (CDotFeatures*)(((bmrm_data_T*)data)->X);
	CLabels* labels = (CLabels*)(((bmrm_data_T*)data)->y);
	int32_t w_dim = ((bmrm_data_T*)data)->w_dim;
	risk_directed(features,labels,SGVector<float64_t>(R,1,false),
	              SGVector<float64_t>(subgrad,w_dim,false),
	              SGVector<float64_t>(W,w_dim,false));
}

void CDirectorRiskFunction::risk_directed(CDotFeatures* features, CLabels* labels, const SGVector<float64_t> R,
                                          const SGVector<float64_t> subgrad, const SGVector<float64_t> W)
{
	SG_NOTIMPLEMENTED;
}
