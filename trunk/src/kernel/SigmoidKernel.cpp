/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SigmoidKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

CSigmoidKernel::CSigmoidKernel(LONG size, DREAL g, DREAL c)
  : CSimpleKernel<DREAL>(size),gamma(g), coef0(c)
{
}

CSigmoidKernel::CSigmoidKernel(CRealFeatures* l, CRealFeatures* r, INT size, DREAL g, DREAL c)
  : CSimpleKernel<DREAL>(size),gamma(g), coef0(c)

{
	init(l,r, true);
}

CSigmoidKernel::~CSigmoidKernel() 
{
	cleanup();
}
  
bool CSigmoidKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSimpleKernel<DREAL>::init(l, r, do_init); 
	return true;
}

void CSigmoidKernel::cleanup()
{
}

bool CSigmoidKernel::load_init(FILE* src)
{
	return false;
}

bool CSigmoidKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CSigmoidKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  ASSERT(alen==blen);

  INT ialen=(int) alen;

#ifndef HAVE_LAPACK
  DREAL result=0;
  {
    for (INT i=0; i<ialen; i++)
      result+=avec[i]*bvec[i];
  }
#else
  INT skip=1;
  DREAL result = cblas_ddot(ialen, avec, skip, bvec, skip);
#endif

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return tanh(gamma*result+coef0);
}
