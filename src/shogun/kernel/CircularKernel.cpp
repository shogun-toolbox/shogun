/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Abhinav Maurya
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 * Copyright (C) 2011 Indian Institute of Technology Bombay
 */

#include <kernel/CircularKernel.h>
#include <mathematics/Math.h>

using namespace shogun;

CCircularKernel::CCircularKernel(): CKernel(0), distance(NULL)
{
	init();
	set_sigma(1.0);
}

CCircularKernel::CCircularKernel(int32_t size, float64_t sig, CDistance* dist)
: CKernel(size), distance(dist)
{
	ASSERT(distance)
	SG_REF(distance);

	set_sigma(sig);
	init();
}

CCircularKernel::CCircularKernel(
	CFeatures *l, CFeatures *r, float64_t sig, CDistance* dist)
: CKernel(10), distance(dist)
{
	ASSERT(distance)
	SG_REF(distance);
	set_sigma(sig);
	init();
	init(l, r);
}

CCircularKernel::~CCircularKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CCircularKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CCircularKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
}

void CCircularKernel::init()
{
	SG_ADD((CSGObject**) &distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
	SG_ADD(&sigma, "sigma", "Sigma kernel parameter.", MS_AVAILABLE);
}

float64_t CCircularKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma)
		return (2/M_PI)*acos(-ds_ratio) - (2/M_PI)*ds_ratio*sqrt(1-ds_ratio*ds_ratio);
	else
		return 0;
}
