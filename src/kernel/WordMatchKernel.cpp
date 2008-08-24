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
#include "kernel/WordMatchKernel.h"
#include "kernel/AvgDiagKernelNormalizer.h"
#include "features/WordFeatures.h"

CWordMatchKernel::CWordMatchKernel(INT size, INT d)
: CSimpleKernel<WORD>(size), degree(d)
{
	set_normalizer(new CAvgDiagKernelNormalizer());
}

CWordMatchKernel::CWordMatchKernel(CWordFeatures* l, CWordFeatures* r, INT d)
: CSimpleKernel<WORD>(10), degree(d)
{
	set_normalizer(new CAvgDiagKernelNormalizer());
	init(l, r);
}

CWordMatchKernel::~CWordMatchKernel()
{
	cleanup();
}

bool CWordMatchKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<WORD>::init(l, r);
	return init_normalizer();
}

bool CWordMatchKernel::load_init(FILE* src)
{
	return false;
}

bool CWordMatchKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CWordMatchKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  ASSERT(alen==blen);

  DREAL sum=0;
  for (INT i=0; i<alen; i++)
	  sum+= (avec[i]==bvec[i]) ? 1 : 0;

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return CMath::pow(sum, degree);
}
