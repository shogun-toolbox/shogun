/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <mathematics/Math.h>
#include <kernel/TensorProductPairKernel.h>
#include <io/SGIO.h>

using namespace shogun;

CTensorProductPairKernel::CTensorProductPairKernel()
: CDotKernel(0), subkernel(NULL)
{
	register_params();
}

CTensorProductPairKernel::CTensorProductPairKernel(int32_t size, CKernel* s)
: CDotKernel(size), subkernel(s)
{
	SG_REF(subkernel);
	register_params();
}

CTensorProductPairKernel::CTensorProductPairKernel(CDenseFeatures<int32_t>* l, CDenseFeatures<int32_t>* r, CKernel* s)
: CDotKernel(10), subkernel(s)
{
	SG_REF(subkernel);
	init(l, r);
	register_params();
}

CTensorProductPairKernel::~CTensorProductPairKernel()
{
	SG_UNREF(subkernel);
	cleanup();
}

bool CTensorProductPairKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	init_normalizer();
	return true;
}

float64_t CTensorProductPairKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	int32_t* avec=((CDenseFeatures<int32_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	int32_t* bvec=((CDenseFeatures<int32_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==2)
	ASSERT(blen==2)

	CKernel* k=subkernel;
	ASSERT(k && k->has_features())

	int32_t a=avec[0];
	int32_t b=avec[1];
	int32_t c=bvec[0];
	int32_t d=bvec[1];

	float64_t result = k->kernel(a,c)*k->kernel(b,d) + k->kernel(a,d)*k->kernel(b,c);

	((CDenseFeatures<int32_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<int32_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void CTensorProductPairKernel::register_params()
{
	SG_ADD((CSGObject**)&subkernel, "subkernel", "the subkernel", MS_AVAILABLE);
}
