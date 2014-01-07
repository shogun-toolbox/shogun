/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <io/SGIO.h>
#include <kernel/string/LocalityImprovedStringKernel.h>
#include <kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <features/StringFeatures.h>

using namespace shogun;

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel()
: CStringKernel<char>()
{
	init();
}

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel(
	int32_t size, int32_t l, int32_t id, int32_t od)
: CStringKernel<char>(size)
{
	init();

	length=l;
	inner_degree=id;
	outer_degree=od;

	SG_DEBUG("LIK with parms: l=%d, id=%d, od=%d created!\n", l, id, od)
}

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t len,
	int32_t id, int32_t od)
: CStringKernel<char>()
{
	init();

	length=len;
	inner_degree=id;
	outer_degree=od;

	SG_DEBUG("LIK with parms: l=%d, id=%d, od=%d created!\n", len, id, od)

	init(l, r);
}

CLocalityImprovedStringKernel::~CLocalityImprovedStringKernel()
{
	cleanup();
}

bool CLocalityImprovedStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l,r);
	return init_normalizer();
}

float64_t CLocalityImprovedStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen && alen>0)

	int32_t i,t;
	float64_t* match=SG_MALLOC(float64_t, alen);

	// initialize match table 1 -> match;  0 -> no match
	for (i = 0; i<alen; i++)
		match[i] = (avec[i] == bvec[i])? 1 : 0;

	float64_t outer_sum = 0;

	for (t = 0; t<alen-length; t++)
	{
		float64_t sum = 0;
		for (i = 0; i<length && t+i+length+1<alen; i++)
			sum += (i+1)*match[t+i]+(length-i)*match[t+i+length+1];
		//add middle element + normalize with sum_i=0^2l+1 i = (2l+1)(l+1)
		float64_t inner_sum = (sum + (length+1)*match[t+length]) / ((2*length+1)*(length+1));
		inner_sum = pow(inner_sum, inner_degree + 1);
		outer_sum += inner_sum;
	}
	SG_FREE(match);

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return pow(outer_sum, outer_degree + 1);
}

void CLocalityImprovedStringKernel::init()
{
	set_normalizer(new CSqrtDiagKernelNormalizer());

	length = 0;
	inner_degree = 0;
	outer_degree = 0;

	SG_ADD(&length, "length", "Window Length.", MS_AVAILABLE);
	SG_ADD(&inner_degree, "inner_degree", "Inner degree.", MS_AVAILABLE);
	SG_ADD(&outer_degree, "outer_degree", "Outer degree.", MS_AVAILABLE);
}
