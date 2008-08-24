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
#include "kernel/PolyMatchStringKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

CPolyMatchStringKernel::CPolyMatchStringKernel(INT size, INT d, bool i)
: CStringKernel<CHAR>(size), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CPolyMatchStringKernel::CPolyMatchStringKernel(
	CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT d, bool i)
: CStringKernel<CHAR>(10), degree(d), inhomogene(i)
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
	CStringKernel<CHAR>::init(l, r);
	return init_normalizer();
}

void CPolyMatchStringKernel::cleanup()
{
	CKernel::cleanup();
}

bool CPolyMatchStringKernel::load_init(FILE *src)
{
	return false;
}

bool CPolyMatchStringKernel::save_init(FILE *dest)
{
	return false;
}

DREAL CPolyMatchStringKernel::compute(INT idx_a, INT idx_b)
{
	INT i, alen, blen, sum;

	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);
	for (i = 0, sum = inhomogene; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			sum++;
	}
	return pow(sum, degree);
}
