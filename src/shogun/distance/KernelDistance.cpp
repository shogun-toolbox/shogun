/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Christian Widmer
 * Copyright (C) 2010 Max-Planck-Society
 */

#include <lib/config.h>

#include <lib/common.h>
#include <io/SGIO.h>
#include <distance/KernelDistance.h>
#include <features/DenseFeatures.h>

using namespace shogun;

CKernelDistance::CKernelDistance() : CDistance()
{
	init();
}

CKernelDistance::CKernelDistance(float64_t w, CKernel* k)
: CDistance()
{
	init();

	kernel=k;
	width=w;
	ASSERT(kernel)
	SG_REF(kernel);
}

CKernelDistance::CKernelDistance(
	CFeatures *l, CFeatures *r, float64_t w , CKernel* k)
: CDistance()
{
	init();

	kernel=k;
	width=w;
	ASSERT(kernel)
	SG_REF(kernel);

	init(l, r);
}

CKernelDistance::~CKernelDistance()
{
	// important to have the cleanup of CDistance first, it calls get_name which
	// uses the distance
	cleanup();
	SG_UNREF(kernel);
}

bool CKernelDistance::init(CFeatures* l, CFeatures* r)
{
	ASSERT(kernel)
	kernel->init(l,r);
	return CDistance::init(l,r);
}

float64_t CKernelDistance::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=kernel->kernel(idx_a, idx_b);
	return exp(-result/width);
}

void CKernelDistance::init()
{
	kernel = NULL;
	width = 0.0;

	m_parameters->add(&width, "width", "Width of RBF Kernel");
	m_parameters->add((CSGObject**) &kernel, "kernel",
					  "Kernel.");
}
