/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/structure/DirectorRiskFunction.h>

using namespace shogun;

CDirectorRiskFunction::CDirectorRiskFunction()
:CRiskFunction()
{
}

CDirectorRiskFunction::~CDirectorRiskFunction()
{
}

	CDirectorRiskData::CDirectorRiskData()
:CRiskData()
{
}

CDirectorRiskData::CDirectorRiskData(
		CDotFeatures    *X,
		CLabels         *y,
		uint32_t        w_dim,
		uint32_t        nFeatures)
:CRiskData()
{
	m_X=X;
	m_y=y;
	m_w_dim=w_dim;
	m_nFeatures=nFeatures;
}

CDirectorRiskData::~CDirectorRiskData()
{
}

void CDirectorRiskFunction::risk(
		void* 				data,
		float64_t* 			R,
		float64_t* 			subgrad,
		float64_t* 			W,
		TMultipleCPinfo* 	info)
{
	CDirectorRiskData* data_struct=(CDirectorRiskData*)data;
	CDotFeatures* features=data_struct->m_X;
	CLabels* labels=data_struct->m_y;
	uint32_t w_dim=data_struct->m_w_dim;

	risk_directed(
			features,
			labels,
			SGVector< float64_t >(R, 1, false),
			SGVector< float64_t >(subgrad, w_dim, false),
			SGVector< float64_t >(W, w_dim, false));
}

void CDirectorRiskFunction::risk_directed(
		CDotFeatures*               features,
		CLabels*                    labels,
		SGVector<float64_t>         R,
		SGVector<float64_t>         subgrad,
		const SGVector<float64_t>   W)
{
	SG_NOTIMPLEMENTED;
}
