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
#include <kernel/string/FixedDegreeStringKernel.h>
#include <kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <features/Features.h>
#include <features/StringFeatures.h>
#include <io/SGIO.h>

using namespace shogun;

void
CFixedDegreeStringKernel::init()
{
	SG_ADD(&degree, "degree", "The degree.", MS_AVAILABLE);
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel()
: CStringKernel<char>(0), degree(0)
{
	init();
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel(int32_t size, int32_t d)
: CStringKernel<char>(size), degree(d)
{
	init();
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t d)
: CStringKernel<char>(10), degree(d)
{
	init();
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
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	// can only deal with strings of same length
	ASSERT(alen==blen)

	int64_t sum = 0;
	for (int32_t i = 0; i<alen-degree+1; i++)
	{
		bool match = true;

		for (int32_t j = i; j<i+degree && match; j++)
			match = avec[j]==bvec[j];
		if (match)
			sum++;
	}
	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	return sum;
}
