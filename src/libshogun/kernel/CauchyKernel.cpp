/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "CauchyKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

CCauchyKernel::CCauchyKernel(): CKernel(0), distance(NULL), sigma(1.0)
{
	SG_UNSTABLE("CCauchyKernel::CCauchyKernel()", "\n");
}

CCauchyKernel::CCauchyKernel(int32_t cache, float64_t sigma, CDistance* dist)
: CKernel(cache), distance(dist), sigma(sigma)
{
	ASSERT(distance);
	SG_REF(distance);
}

CCauchyKernel::CCauchyKernel(CFeatures *l, CFeatures *r, float64_t sigma, CDistance* dist)
: CKernel(10), distance(dist), sigma(sigma)
{
	ASSERT(distance);
	SG_REF(distance);
	init(l, r);
}

CCauchyKernel::~CCauchyKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CCauchyKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CCauchyKernel::init()
{
	m_parameters->add(&sigma, "sigma", "Sigma kernel parameter.");
}

float64_t CCauchyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return 1.0/(1.0+dist*dist/sigma);
}
