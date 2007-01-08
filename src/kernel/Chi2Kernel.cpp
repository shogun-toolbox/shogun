/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/Chi2Kernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CChi2Kernel::CChi2Kernel(LONG size)
  : CSimpleKernel<DREAL>(size)
{
}

CChi2Kernel::CChi2Kernel(CRealFeatures* l, CRealFeatures* r, INT size)
  : CSimpleKernel<DREAL>(size)
{
	init(l,r, true);
}

CChi2Kernel::~CChi2Kernel() 
{
	cleanup();
}
  
bool CChi2Kernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CSimpleKernel<DREAL>::init(l,r,do_init);
	initialized = true ;
	return result;
}

void CChi2Kernel::cleanup()
{
	initialized = false ;
}

bool CChi2Kernel::load_init(FILE* src)
{
	return false;
}

bool CChi2Kernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CChi2Kernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);

	DREAL result=0;
	for (INT i=0; i<alen; i++)
	{
		DREAL n=avec[i]-bvec[i];
		DREAL d=avec[i]+bvec[i];
		result+=n*n/d;
	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
