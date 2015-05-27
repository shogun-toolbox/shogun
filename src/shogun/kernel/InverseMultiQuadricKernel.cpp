/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Abhinav Maurya
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 Indian Institute of Technology Bombay
 */

#include <shogun/kernel/InverseMultiQuadricKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CInverseMultiQuadricKernel::CInverseMultiQuadricKernel(): CKernel(0), distance(NULL), coef(0.0001)
{
	init();
}

CInverseMultiQuadricKernel::CInverseMultiQuadricKernel(int32_t cache, float64_t coefficient, distance::CDistance* dist)
: CKernel(cache), distance(dist), coef(coefficient)
{
	SG_REF(distance);
	init();
}

CInverseMultiQuadricKernel::CInverseMultiQuadricKernel(CFeatures *l, CFeatures *r, float64_t coefficient, distance::CDistance* dist)
: CKernel(10), distance(dist), coef(coefficient)
{
	SG_REF(distance);
	init();
	init(l, r);
}

CInverseMultiQuadricKernel::~CInverseMultiQuadricKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CInverseMultiQuadricKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CInverseMultiQuadricKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
}

void CInverseMultiQuadricKernel::init()
{
	SG_ADD(&coef, "coef", "Kernel Coefficient.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}

float64_t CInverseMultiQuadricKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return 1/sqrt(dist*dist + coef*coef);
}
