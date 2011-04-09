/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gaussian Kernel used as template, attribution:
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Exponential Kernel
 * Written (W) 2011 Justin Patera
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "base/Parameter.h"
#include "kernel/ExponentialKernel.h"
#include "features/DotFeatures.h"
#include "lib/io.h"

using namespace shogun;

CExponentialKernel::CExponentialKernel()
	: CDotKernel()
{
	init();
}


CExponentialKernel::CExponentialKernel(int32_t size, float64_t w)
: CDotKernel(size)
{
	init();
	width=w;
	ASSERT(distance);
	SG_REF(distance);
}

CExponentialKernel::CExponentialKernel(
	CDotFeatures* l, CDotFeatures* r, float64_t w, int32_t size)
: CDotKernel(size)
{
	init();
	width=w;
	ASSERT(distance);
	SG_REF(distance);
	init(l,r);
}

CExponentialKernel::~CExponentialKernel()
{
	cleanup();
	SG_UNREF(distance);
}

void CExponentialKernel::cleanup()
{
	CKernel::cleanup();
}

bool CExponentialKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	distance->init(l, r);
	return init_normalizer();
}

float64_t CExponentialKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	return exp(-dist/width);
}

void CExponentialKernel::load_serializable_post(void) throw (ShogunException)
{
	CKernel::load_serializable_post();
}


void CExponentialKernel::init()
{
	width=1;
	m_parameters->add(&width, "width", "Kernel width.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}
