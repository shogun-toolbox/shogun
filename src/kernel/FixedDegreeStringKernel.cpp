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
#include "kernel/FixedDegreeStringKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CFixedDegreeStringKernel::CFixedDegreeStringKernel(INT size, INT d)
: CStringKernel<CHAR>(size), degree(d)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel(
	CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT d)
: CStringKernel<CHAR>(10), degree(d)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
}

CFixedDegreeStringKernel::~CFixedDegreeStringKernel()
{
	cleanup();
}

bool CFixedDegreeStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<CHAR>::init(l, r);
	return init_normalizer();
}

void CFixedDegreeStringKernel::cleanup()
{
	CKernel::cleanup();
}

bool CFixedDegreeStringKernel::load_init(FILE* src)
{
	return false;
}

bool CFixedDegreeStringKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CFixedDegreeStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen==blen);

	LONG sum = 0;
	for (INT i = 0; i<alen-degree+1; i++)
	{
		bool match = true;

		for (INT j = i; j<i+degree && match; j++)
			match = avec[j]==bvec[j];
		if (match)
			sum++;
	}
	return sum;
}
