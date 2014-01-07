/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <lib/common.h>
#include <kernel/SigmoidKernel.h>

using namespace shogun;

CSigmoidKernel::CSigmoidKernel() : CDotKernel()
{
	init();
}

CSigmoidKernel::CSigmoidKernel(int32_t size, float64_t g, float64_t c)
: CDotKernel(size)
{
	init();

	gamma=g;
	coef0=c;
}

CSigmoidKernel::CSigmoidKernel(
	CDotFeatures* l, CDotFeatures* r, int32_t size, float64_t g, float64_t c)
: CDotKernel(size)
{
	init();

	gamma=g;
	coef0=c;

	init(l,r);
}

CSigmoidKernel::~CSigmoidKernel()
{
	cleanup();
}

void CSigmoidKernel::cleanup()
{
}

bool CSigmoidKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	return init_normalizer();
}

void CSigmoidKernel::init()
{
	gamma=0.0;
	coef0=0.0;

	SG_ADD(&gamma, "gamma", "Gamma.", MS_AVAILABLE);
	SG_ADD(&coef0, "coef0", "Coefficient 0.", MS_AVAILABLE);
}
