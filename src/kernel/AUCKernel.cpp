/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "kernel/AUCKernel.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

CAUCKernel::CAUCKernel(INT size, CKernel* s)
	: CSimpleKernel<WORD>(size),subkernel(s)
{
}

CAUCKernel::CAUCKernel(CWordFeatures* l, CWordFeatures* r, CKernel* s)
	: CSimpleKernel<WORD>(10),subkernel(s)
{
	init(l, r);
}

CAUCKernel::~CAUCKernel()
{
	cleanup();
}
  
bool CAUCKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<WORD>::init(l, r);
	return true;
}


void CAUCKernel::cleanup()
{
}

bool CAUCKernel::load_init(FILE* src)
{
	return false;
}

bool CAUCKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CAUCKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  ASSERT(alen==2);
  ASSERT(blen==2);

  ASSERT(subkernel && subkernel->has_features());

  DREAL k11,k12,k21,k22 ;
  INT idx_a1=avec[0], idx_a2=avec[1], idx_b1=bvec[0], idx_b2=bvec[1] ;
  
  k11 = subkernel->kernel(idx_a1,idx_b1) ;
  k12 = subkernel->kernel(idx_a1,idx_b2) ;
  k21 = subkernel->kernel(idx_a2,idx_b1) ;  
  k22 = subkernel->kernel(idx_a2,idx_b2) ;

  DREAL result = k11+k22-k21-k12 ;

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
