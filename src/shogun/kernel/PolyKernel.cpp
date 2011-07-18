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

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/io.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/SqrtDiagKernelNormalizer.h>
#include <shogun/features/DotFeatures.h>

using namespace shogun;

CPolyKernel::CPolyKernel(void)
: CDotKernel(0), degree(0), inhomogene(false)
{
	init();

	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CPolyKernel::CPolyKernel(int32_t size, int32_t d, bool i)
: CDotKernel(size), degree(d), inhomogene(i)
{
	init();

	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CPolyKernel::CPolyKernel(
	CDotFeatures* l, CDotFeatures* r, int32_t d, bool i, int32_t size)
: CDotKernel(size), degree(d), inhomogene(i)
{
	init();

	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l,r);
}

CPolyKernel::~CPolyKernel()
{
	cleanup();
}

bool CPolyKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l,r);
	return init_normalizer();
}

void CPolyKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CPolyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=CDotKernel::compute(idx_a, idx_b);

	if (inhomogene)
		result+=1;

	return CMath::pow(result, degree);
}

void CPolyKernel::init()
{
	m_parameters->add(&degree, "degree");
	m_parameters->add(&inhomogene, "inhomogene",
					  "If kernel is inhomogeneous.");
}

