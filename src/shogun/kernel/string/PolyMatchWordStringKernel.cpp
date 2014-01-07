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
#include <kernel/string/PolyMatchWordStringKernel.h>
#include <kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <features/Features.h>
#include <features/StringFeatures.h>

using namespace shogun;

CPolyMatchWordStringKernel::CPolyMatchWordStringKernel()
: CStringKernel<uint16_t>()
{
	init();
}

CPolyMatchWordStringKernel::CPolyMatchWordStringKernel(int32_t size, int32_t d, bool i)
: CStringKernel<uint16_t>(size)
{
	init();

	degree=d;
	inhomogene=i;
}

CPolyMatchWordStringKernel::CPolyMatchWordStringKernel(
	CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, int32_t d, bool i)
: CStringKernel<uint16_t>()
{
	init();

	degree=d;
	inhomogene=i;

	init(l, r);
}

CPolyMatchWordStringKernel::~CPolyMatchWordStringKernel()
{
	cleanup();
}

bool CPolyMatchWordStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<uint16_t>::init(l,r);
	return init_normalizer();
}

void CPolyMatchWordStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CPolyMatchWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=((CStringFeatures<uint16_t>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=((CStringFeatures<uint16_t>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)

	int32_t sum=0;

	for (int32_t i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	if (inhomogene)
		sum+=1;

	float64_t result=sum;

	for (int32_t j=1; j<degree; j++)
		result*=sum;

	((CStringFeatures<uint16_t>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<uint16_t>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

void CPolyMatchWordStringKernel::init()
{
	degree=0;
	inhomogene=false;
	set_normalizer(new CSqrtDiagKernelNormalizer());

	SG_ADD(&degree, "degree", "Degree of poly-kernel.", MS_AVAILABLE);
	SG_ADD(&inhomogene, "inhomogene", "True for inhomogene poly-kernel.",
	    MS_NOT_AVAILABLE);
}
