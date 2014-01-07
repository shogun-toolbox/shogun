/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Jakub Jirku
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <kernel/LogKernel.h>
#include <mathematics/Math.h>

using namespace shogun;

CLogKernel::CLogKernel(): CKernel(0), m_distance(NULL), m_degree(1.8)
{
	init();
}

CLogKernel::CLogKernel(int32_t cache, float64_t degree, CDistance* dist)
: CKernel(cache), m_distance(dist), m_degree(degree)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
}

CLogKernel::CLogKernel(CFeatures *l, CFeatures *r, float64_t degree, CDistance* dist)
: CKernel(10), m_distance(dist), m_degree(degree)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
	init(l, r);
}

CLogKernel::~CLogKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

bool CLogKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CKernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void CLogKernel::init()
{
	SG_ADD(&m_degree, "degree", "Degree kernel parameter.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}

float64_t CLogKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	float64_t temp = pow(dist, m_degree);
	temp = log(temp + 1);
	return -temp;
}
