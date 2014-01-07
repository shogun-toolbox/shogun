/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <lib/common.h>
#include <kernel/JensenShannonKernel.h>
#include <features/Features.h>
#include <io/SGIO.h>

using namespace shogun;

CJensenShannonKernel::CJensenShannonKernel()
: CDotKernel(0)
{
}

CJensenShannonKernel::CJensenShannonKernel(int32_t size)
: CDotKernel(size)
{
}

CJensenShannonKernel::CJensenShannonKernel(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r, int32_t size)
: CDotKernel(size)
{
	init(l,r);
}

CJensenShannonKernel::~CJensenShannonKernel()
{
	cleanup();
}

bool CJensenShannonKernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CDotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t CJensenShannonKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;

	/* calcualte Jensen-Shannon kernel */
	for (int32_t i=0; i<alen; i++) {
		float64_t a_i = 0, b_i = 0;
		float64_t ab = avec[i]+bvec[i];
		if (avec[i] != 0)
			a_i = avec[i] * CMath::log2(ab/avec[i]);
		if (bvec[i] != 0)
			b_i = bvec[i] * CMath::log2(ab/bvec[i]);

		result += 0.5*(a_i + b_i);
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

