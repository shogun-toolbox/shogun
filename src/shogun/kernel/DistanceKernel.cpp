/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Gehl
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/DistanceKernel.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CDistanceKernel::CDistanceKernel()
: CKernel(0), distance(NULL), width(0.0)
{
	register_params();
}

CDistanceKernel::CDistanceKernel(int32_t size, float64_t w, CDistance* d)
: CKernel(size), distance(d)
{
	ASSERT(distance)
	set_width(w);
	SG_REF(distance);
	register_params();
}

CDistanceKernel::CDistanceKernel(
	CFeatures *l, CFeatures *r, float64_t w , CDistance* d)
: CKernel(10), distance(d)
{
	set_width(w);
	ASSERT(distance)
	SG_REF(distance);
	init(l, r);
	register_params();
}

CDistanceKernel::~CDistanceKernel()
{
	// important to have the cleanup of CKernel first, it calls get_name which
	// uses the distance
	cleanup();
	SG_UNREF(distance);
}

bool CDistanceKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t CDistanceKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=distance->distance(idx_a, idx_b);
	return exp(-result/width);
}

void CDistanceKernel::register_params()
{
	SG_ADD(&width, "width", "Kernel width.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}
