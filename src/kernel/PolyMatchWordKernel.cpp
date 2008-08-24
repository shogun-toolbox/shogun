/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/PolyMatchWordKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/Features.h"
#include "features/WordFeatures.h"

CPolyMatchWordKernel::CPolyMatchWordKernel(INT size, INT d, bool i)
: CSimpleKernel<WORD>(size),degree(d),inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CPolyMatchWordKernel::CPolyMatchWordKernel(
	CWordFeatures* l, CWordFeatures* r, INT d, bool i)
: CSimpleKernel<WORD>(10),degree(d),inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
}

CPolyMatchWordKernel::~CPolyMatchWordKernel()
{
	cleanup();
}

bool CPolyMatchWordKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<WORD>::init(l,r);
	return init_normalizer();
}

void CPolyMatchWordKernel::cleanup()
{
	CKernel::cleanup();
}

bool CPolyMatchWordKernel::load_init(FILE* src)
{
	return false;
}

bool CPolyMatchWordKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CPolyMatchWordKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	//fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
	WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	INT sum=0;

	for (INT i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	if (inhomogene)
		sum+=1;

	DREAL result=sum;

	for (INT j=1; j<degree; j++)
		result*=sum;

	((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
