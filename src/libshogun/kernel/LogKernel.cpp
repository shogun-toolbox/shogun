/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Jakub Jirku
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "LogKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

CLogKernel::CLogKernel(): CKernel(0), distance(NULL), degree(1.8)
{
	init();
}

CLogKernel::CLogKernel(int32_t cache, float64_t degree, CDistance* dist)
: CKernel(cache), distance(dist), degree(degree)
{
	init();
	ASSERT(distance);
	SG_REF(distance);
}

CLogKernel::CLogKernel(CFeatures *l, CFeatures *r, float64_t degree, CDistance* dist)
: CKernel(10), distance(dist), degree(degree)
{
	init();
	ASSERT(distance);
	SG_REF(distance);
	init(l, r);
}

CLogKernel::~CLogKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CLogKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CLogKernel::init()
{
	m_parameters->add(&degree, "degree", "Degree kernel parameter.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}

float64_t CLogKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);	
	float64_t temp = pow(dist, degree);
	temp = log(temp + 1);
	return -temp;
}
