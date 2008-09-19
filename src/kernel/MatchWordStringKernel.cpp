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

CMatchWordStringKernel::CMatchWordStringKernel(INT size, INT d)
: CStringKernel<WORD>(size), degree(d)
{
	set_normalizer(new CAvgDiagKernelNormalizer());
}

CMatchWordStringKernel::CMatchWordStringKernel(CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, INT d)
: CStringKernel<WORD>(10), degree(d)
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
	CStringKernel<WORD>::init(l, r);
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
  
DREAL CMatchWordStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
	WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);
	// can only deal with strings of same length
	ASSERT(alen==blen);

	DREAL sum=0;
	for (INT i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	return CMath::pow(sum, degree);
}
