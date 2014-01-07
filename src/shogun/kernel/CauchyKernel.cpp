/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <kernel/CauchyKernel.h>
#include <mathematics/Math.h>

using namespace shogun;

CCauchyKernel::CCauchyKernel(): CKernel(0), m_distance(NULL), m_sigma(1.0)
{
	init();
}

CCauchyKernel::CCauchyKernel(int32_t cache, float64_t sigma, CDistance* dist)
: CKernel(cache), m_distance(dist), m_sigma(sigma)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
}

CCauchyKernel::CCauchyKernel(CFeatures *l, CFeatures *r, float64_t sigma, CDistance* dist)
: CKernel(10), m_distance(dist), m_sigma(sigma)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
	init(l, r);
}

CCauchyKernel::~CCauchyKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

bool CCauchyKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CKernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void CCauchyKernel::init()
{
	SG_ADD(&m_sigma, "sigma", "Sigma kernel parameter.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}

float64_t CCauchyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	return 1.0/(1.0+dist*dist/m_sigma);
}
