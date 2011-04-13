/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Joanna Stocka
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "MultiquadricKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

CMultiquadricKernel::CMultiquadricKernel(): CKernel(0), distance(NULL), coef(0.0001)
{
	init();	
}

CMultiquadricKernel::CMultiquadricKernel(int32_t cache, float64_t coef, CDistance* dist)
: CKernel(cache), distance(dist), coef(coef)
{
	ASSERT(distance);
	SG_REF(distance);
	init();
}

CMultiquadricKernel::CMultiquadricKernel(CFeatures *l, CFeatures *r, float64_t coef, CDistance* dist)
: CKernel(10), distance(dist), coef(coef)
{
	ASSERT(distance);
	SG_REF(distance);
	init(l, r);
	init();
}

CMultiquadricKernel::~CMultiquadricKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CMultiquadricKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t CMultiquadricKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return sqrt(dist*dist + coef*coef);
}

void CMultiquadricKernel::init()
{
	m_parameters->add(&coef, "coef", "Kernel coefficient.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}
