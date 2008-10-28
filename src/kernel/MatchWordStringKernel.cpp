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
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "kernel/MatchWordStringKernel.h"
#include "kernel/AvgDiagKernelNormalizer.h"
#include "features/StringFeatures.h"

CMatchWordStringKernel::CMatchWordStringKernel(int32_t size, int32_t d)
: CStringKernel<uint16_t>(size), degree(d)
{
	set_normalizer(new CAvgDiagKernelNormalizer());
}

CMatchWordStringKernel::CMatchWordStringKernel(CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, int32_t d)
: CStringKernel<uint16_t>(10), degree(d)
{
	set_normalizer(new CAvgDiagKernelNormalizer());
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

bool CMatchWordStringKernel::load_init(FILE* src)
{
	return false;
}

bool CMatchWordStringKernel::save_init(FILE* dest)
{
	return false;
}
  
float64_t CMatchWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;

	uint16_t* avec=((CStringFeatures<uint16_t>*) lhs)->get_feature_vector(idx_a, alen);
	uint16_t* bvec=((CStringFeatures<uint16_t>*) rhs)->get_feature_vector(idx_b, blen);
	// can only deal with strings of same length
	ASSERT(alen==blen);

	float64_t sum=0;
	for (int32_t i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	return CMath::pow(sum, degree);
}
