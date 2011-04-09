/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Based on GaussianKernel, Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "SphericalKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

CSphericalKernel::CSphericalKernel(void): CKernel(0), distance(NULL), sigma(1.0)
{
	SG_UNSTABLE("CSphericalKernel::CSphericalKernel(void)", "\n");
}

CSphericalKernel::CSphericalKernel(int32_t size, float64_t sigma, CDistance* dist)
: CKernel(size), distance(dist)
{
	ASSERT(distance);
	init();
	set_sigma(sigma);
	SG_REF(distance);
}

CSphericalKernel::CSphericalKernel(
	CFeatures *l, CFeatures *r, float64_t sigma, CDistance* dist)
: CKernel(10), distance(dist)
{
  	ASSERT(distance);
	init();
	set_sigma(sigma);
	SG_REF(distance);
	init(l, r);
}

CSphericalKernel::~CSphericalKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CSphericalKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CSphericalKernel::init()
{
 	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
	m_parameters->add(&sigma, "sigma", "Sigma kernel parameter.");
}

float64_t CSphericalKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma) 
		return 1.0-1.5*(ds_ratio)+0.5*(ds_ratio*ds_ratio*ds_ratio);
	else
		return 0;
}
