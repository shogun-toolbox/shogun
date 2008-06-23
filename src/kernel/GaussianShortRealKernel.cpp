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
#include "kernel/GaussianShortRealKernel.h"
#include "features/Features.h"
#include "features/ShortRealFeatures.h"
#include "lib/io.h"

CGaussianShortRealKernel::CGaussianShortRealKernel(INT size, DREAL w)
: CSimpleKernel<DREAL>(size), width(w)
{
}

CGaussianShortRealKernel::CGaussianShortRealKernel(
	CShortRealFeatures* l, CShortRealFeatures* r, DREAL w, INT size)
: CSimpleKernel<DREAL>(size), width(w)
{
	init(l,r);
}

CGaussianShortRealKernel::~CGaussianShortRealKernel()
{
}

bool CGaussianShortRealKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<DREAL>::init(l, r);
	return true;
}

bool CGaussianShortRealKernel::load_init(FILE* src)
{
	return false;
}

bool CGaussianShortRealKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CGaussianShortRealKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	SHORTREAL* avec=((CShortRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	SHORTREAL* bvec=((CShortRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);

	DREAL result=0;
	for (INT i=0; i<alen; i++)
		result+=CMath::sq(avec[i]-bvec[i]);

	result=exp(-result/width);

	((CShortRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CShortRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
