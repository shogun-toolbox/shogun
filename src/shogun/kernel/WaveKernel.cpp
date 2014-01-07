/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <kernel/WaveKernel.h>
#include <mathematics/Math.h>

using namespace shogun;

CWaveKernel::CWaveKernel(): CKernel(0), m_distance(NULL), m_theta(1.0)
{
	init();
}

CWaveKernel::CWaveKernel(int32_t cache, float64_t theta, CDistance* dist)
: CKernel(cache), m_distance(dist), m_theta(theta)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
}

CWaveKernel::CWaveKernel(CFeatures *l, CFeatures *r, float64_t theta, CDistance* dist)
: CKernel(10), m_distance(dist), m_theta(theta)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
	init(l, r);
}

CWaveKernel::~CWaveKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

bool CWaveKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CKernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void CWaveKernel::init()
{
	SG_ADD(&m_theta, "theta", "Theta kernel parameter.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}

float64_t CWaveKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);

	if (dist==0.0)
		return 1.0;

	return (m_theta/dist)*sin(dist/m_theta);
}
