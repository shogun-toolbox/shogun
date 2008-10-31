/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/lapack.h"
#include "kernel/SigmoidKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CSigmoidKernel::CSigmoidKernel(int32_t size, float64_t g, float64_t c)
: CSimpleKernel<float64_t>(size),gamma(g), coef0(c)
{
}

CSigmoidKernel::CSigmoidKernel(
	CRealFeatures* l, CRealFeatures* r, int32_t size, float64_t g, float64_t c)
: CSimpleKernel<float64_t>(size),gamma(g), coef0(c)
{
	init(l,r);
}

CSigmoidKernel::~CSigmoidKernel()
{
	cleanup();
}

bool CSigmoidKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<float64_t>::init(l, r);
	return init_normalizer();
}

void CSigmoidKernel::cleanup()
{
}

bool CSigmoidKernel::load_init(FILE* src)
{
	return false;
}

bool CSigmoidKernel::save_init(FILE* dest)
{
	return false;
}

float64_t CSigmoidKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);

#ifndef HAVE_LAPACK
	float64_t result=0;
	{
		for (int32_t i=0; i<alen; i++)
			result+=avec[i]*bvec[i];
	}
#else
	int skip=1; /* calling external lib */
	float64_t result = cblas_ddot(
		(int) alen, (double*) avec, skip, (double*) bvec, skip);
#endif

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return tanh(gamma*result+coef0);
}
