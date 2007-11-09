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
#include "kernel/GaussianKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CGaussianKernel::CGaussianKernel(INT size, DREAL w)
  : CSimpleKernel<DREAL>(size),width(w)
{
}

CGaussianKernel::CGaussianKernel(CRealFeatures* l, CRealFeatures* r, DREAL w, INT size)
  : CSimpleKernel<DREAL>(size),width(w)
{
	init(l,r);
}

CGaussianKernel::~CGaussianKernel()
{
}
  
bool CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<DREAL>::init(l, r);
	return true;
}

void CGaussianKernel::cleanup()
{
}

bool CGaussianKernel::load_init(FILE* src)
{
	return false;
}

bool CGaussianKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CGaussianKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  DREAL* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  DREAL* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  ASSERT(alen==blen);

  INT ialen=(int) alen;

  DREAL result=0;
  for (INT i=0; i<ialen; i++)
	  result+=(avec[i]-bvec[i])*(avec[i]-bvec[i]);

  result=exp(-result/width);

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
