/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/PolyMatchStringKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

using namespace shogun;

CPolyMatchStringKernel::CPolyMatchStringKernel(int32_t size, int32_t d, bool i)
: CStringKernel<char>(size), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CPolyMatchStringKernel::CPolyMatchStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t d, bool i)
: CStringKernel<char>(10), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
}

CPolyMatchStringKernel::~CPolyMatchStringKernel()
{
	cleanup();
}

bool CPolyMatchStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CPolyMatchStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CPolyMatchStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t i, alen, blen, sum;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);
	for (i = 0, sum = inhomogene; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			sum++;
	}
	return CMath::pow((float64_t) sum, degree);
}
