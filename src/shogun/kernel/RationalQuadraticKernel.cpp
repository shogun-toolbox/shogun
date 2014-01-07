/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Rafał Surowiecki
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <kernel/RationalQuadraticKernel.h>
#include <mathematics/Math.h>

using namespace shogun;

CRationalQuadraticKernel::CRationalQuadraticKernel(): CKernel(0), m_distance(NULL), m_coef(0.001)
{
	init();
}

CRationalQuadraticKernel::CRationalQuadraticKernel(int32_t cache, float64_t coef, CDistance* distance)
: CKernel(cache), m_distance(distance), m_coef(coef)
{
	ASSERT(m_distance)
	SG_REF(m_distance);
	init();
}

CRationalQuadraticKernel::CRationalQuadraticKernel(CFeatures *l, CFeatures *r, float64_t coef, CDistance* dist)
: CKernel(10), m_distance(dist), m_coef(coef)
{
	ASSERT(m_distance)
	SG_REF(m_distance);
	init();
	init(l, r);
}

CRationalQuadraticKernel::~CRationalQuadraticKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

bool CRationalQuadraticKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CKernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

float64_t CRationalQuadraticKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	float64_t pDist = dist * dist;
	return 1-pDist/(pDist+m_coef);
}

void CRationalQuadraticKernel::init()
{
	SG_ADD(&m_coef, "coef", "Kernel coefficient.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}

