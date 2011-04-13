/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Rafał Surowiecki
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "RationalQuadraticKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

CRationalQuadraticKernel::CRationalQuadraticKernel(): CKernel(0), distance(NULL), coef(0.001)
{
	init();
}

CRationalQuadraticKernel::CRationalQuadraticKernel(int32_t cache, float64_t coef, CDistance* dist)
: CKernel(cache), distance(dist), coef(coef)
{
	ASSERT(distance);
	SG_REF(distance);
	init();
}

CRationalQuadraticKernel::CRationalQuadraticKernel(CFeatures *l, CFeatures *r, float64_t coef, CDistance* dist)
: CKernel(10), distance(dist), coef(coef)
{
	ASSERT(distance);
	SG_REF(distance);
	init();
	init(l, r);
}

CRationalQuadraticKernel::~CRationalQuadraticKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CRationalQuadraticKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t CRationalQuadraticKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	float64_t pDist = dist * dist;
	return 1-pDist/(pDist+coef);
}

void CRationalQuadraticKernel::init()
{
	m_parameters->add(&coef, "coef", "Kernel coefficient.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}

