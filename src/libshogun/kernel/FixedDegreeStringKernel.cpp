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
#include "kernel/FixedDegreeStringKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CFixedDegreeStringKernel::CFixedDegreeStringKernel(int32_t size, int32_t d)
: CStringKernel<char>(size), degree(d)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t d)
: CStringKernel<char>(10), degree(d)
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
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CFixedDegreeStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CFixedDegreeStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen==blen);

	int64_t sum = 0;
	for (int32_t i = 0; i<alen-degree+1; i++)
	{
		bool match = true;

		for (int32_t j = i; j<i+degree && match; j++)
			match = avec[j]==bvec[j];
		if (match)
			sum++;
	}
	return sum;
}
