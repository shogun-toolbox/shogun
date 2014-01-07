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

#include <lib/common.h>
#include <base/Parameter.h>
#include <kernel/ExponentialKernel.h>
#include <features/DotFeatures.h>
#include <io/SGIO.h>

using namespace shogun;

CExponentialKernel::CExponentialKernel()
	: CDotKernel(), m_distance(NULL), m_width(1)
{
	init();
}

CExponentialKernel::CExponentialKernel(
	CDotFeatures* l, CDotFeatures* r, float64_t width, CDistance* distance, int32_t size)
: CDotKernel(size), m_distance(distance), m_width(width)
{
	init();
	ASSERT(distance)
	SG_REF(distance);
	init(l,r);
}

CExponentialKernel::~CExponentialKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

void CExponentialKernel::cleanup()
{
	CKernel::cleanup();
}

bool CExponentialKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CDotKernel::init(l, r);
	m_distance->init(l, r);
	return init_normalizer();
}

float64_t CExponentialKernel::compute(int32_t idx_a, int32_t idx_b)
{
	ASSERT(m_distance)
	float64_t dist=m_distance->distance(idx_a, idx_b);
	return exp(-dist/m_width);
}

void CExponentialKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
}


void CExponentialKernel::init()
{
	SG_ADD(&m_width, "width", "Kernel width.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}
