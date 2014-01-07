/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <io/SGIO.h>
#include <kernel/string/PolyMatchStringKernel.h>
#include <kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <features/Features.h>
#include <features/StringFeatures.h>

using namespace shogun;

CPolyMatchStringKernel::CPolyMatchStringKernel()
: CStringKernel<char>()
{
	init();
}

CPolyMatchStringKernel::CPolyMatchStringKernel(int32_t size, int32_t d, bool i)
: CStringKernel<char>(size)
{
	init();

	degree=d;
	inhomogene=i;
}

CPolyMatchStringKernel::CPolyMatchStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t d, bool i)
: CStringKernel<char>(10)
{
	init();

	degree=d;
	inhomogene=i;

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
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)
	for (i = 0, sum = inhomogene; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			sum++;
	}
	float64_t result = ((float64_t) sum);

	if (rescaling)
		result/=alen;

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return CMath::pow(result , degree);
}

void CPolyMatchStringKernel::init()
{
	degree=0;
	inhomogene=false;
	rescaling=false;
	set_normalizer(new CSqrtDiagKernelNormalizer());

	SG_ADD(&degree, "degree", "Degree of poly-kernel.", MS_AVAILABLE);
	SG_ADD(&inhomogene, "inhomogene", "True for inhomogene poly-kernel.",
	    MS_NOT_AVAILABLE);
	SG_ADD(&rescaling, "rescaling",
	    "True to rescale kernel with string length.", MS_AVAILABLE);
}
