/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <kernel/GaussianShortRealKernel.h>
#include <features/Features.h>
#include <io/SGIO.h>

using namespace shogun;

CGaussianShortRealKernel::CGaussianShortRealKernel()
: CDotKernel(0), width(0.0)
{
	register_params();
}

CGaussianShortRealKernel::CGaussianShortRealKernel(int32_t size, float64_t w)
: CDotKernel(size), width(w)
{
	register_params();
}

CGaussianShortRealKernel::CGaussianShortRealKernel(
	CDenseFeatures<float32_t>* l, CDenseFeatures<float32_t>* r, float64_t w, int32_t size)
: CDotKernel(size), width(w)
{
	init(l,r);
	register_params();
}

CGaussianShortRealKernel::~CGaussianShortRealKernel()
{
}

bool CGaussianShortRealKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	return init_normalizer();
}

float64_t CGaussianShortRealKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float32_t* avec=((CDenseFeatures<float32_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float32_t* bvec=((CDenseFeatures<float32_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
		result+=CMath::sq(avec[i]-bvec[i]);

	result=exp(-result/width);

	((CDenseFeatures<float32_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float32_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void CGaussianShortRealKernel::register_params()
{
	SG_ADD(&width, "width", "kernel width", MS_AVAILABLE);
}
