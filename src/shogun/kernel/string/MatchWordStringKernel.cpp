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
#include <mathematics/Math.h>
#include <io/SGIO.h>
#include <kernel/string/MatchWordStringKernel.h>
#include <kernel/normalizer/AvgDiagKernelNormalizer.h>
#include <features/StringFeatures.h>

using namespace shogun;

CMatchWordStringKernel::CMatchWordStringKernel() : CStringKernel<uint16_t>()
{
	init();
}

CMatchWordStringKernel::CMatchWordStringKernel(int32_t size, int32_t d)
: CStringKernel<uint16_t>(size)
{
	init();
	degree=d;
}

CMatchWordStringKernel::CMatchWordStringKernel(
		CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, int32_t d)
: CStringKernel<uint16_t>()
{
	init();
	degree=d;
	init(l, r);
}

CMatchWordStringKernel::~CMatchWordStringKernel()
{
	cleanup();
}

bool CMatchWordStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<uint16_t>::init(l, r);
	return init_normalizer();
}

float64_t CMatchWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=((CStringFeatures<uint16_t>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=((CStringFeatures<uint16_t>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)

	float64_t sum=0;
	for (int32_t i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	((CStringFeatures<uint16_t>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<uint16_t>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	return CMath::pow(sum, degree);
}

void CMatchWordStringKernel::init()
{
	degree=0;
	set_normalizer(new CAvgDiagKernelNormalizer());
	SG_ADD(&degree, "degree", "Degree of poly kernel", MS_AVAILABLE);
}
