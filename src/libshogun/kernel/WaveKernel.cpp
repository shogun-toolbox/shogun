/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "WaveKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

CWaveKernel::CWaveKernel(): CKernel(0), distance(NULL), theta(1.0)
{
	init();
}

CWaveKernel::CWaveKernel(int32_t cache, float64_t theta, CDistance* dist)
: CKernel(cache), distance(dist), theta(theta)
{
	init();
	ASSERT(distance);
	SG_REF(distance);
}

CWaveKernel::CWaveKernel(CFeatures *l, CFeatures *r, float64_t theta, CDistance* dist)
: CKernel(10), distance(dist), theta(theta)
{
	init();
	ASSERT(distance);
	SG_REF(distance);
	init(l, r);
}

CWaveKernel::~CWaveKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CWaveKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CWaveKernel::init()
{
	m_parameters->add(&theta, "theta", "Theta kernel parameter.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}

float64_t CWaveKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	if (dist==0.0) return 1.0;
	return (theta/dist)*sin(dist/theta);
}
