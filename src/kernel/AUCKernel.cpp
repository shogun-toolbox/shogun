/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/Mathmatics.h"
#include "kernel/AUCKernel.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

CAUCKernel::CAUCKernel(INT size, CKernel * subkernel_)
	: CSimpleKernel<WORD>(size),subkernel(subkernel_)
{
}

CAUCKernel::~CAUCKernel() 
{
	cleanup();
}
  
bool CAUCKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSimpleKernel<WORD>::init(l, r, do_init); 
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

  ASSERT(subkernel!=NULL) ;
  DREAL k11,k12,k21,k22 ;
  INT idx_a1=avec[0], idx_a2=avec[1], idx_b1=bvec[0], idx_b2=bvec[1] ;
  
  k11 = subkernel->kernel(idx_a1,idx_b1) ;
  k12 = subkernel->kernel(idx_a1,idx_b2) ;
  k21 = subkernel->kernel(idx_a2,idx_b1) ;  
  k22 = subkernel->kernel(idx_a2,idx_b2) ;

  DREAL result = k11+k22-k21-k12 ;

  //CIO::message(M_DEBUG, "k(%i,%i)=%1.2f = k(%i,%i)+k(%i,%i)-k(%i,%i)-k(%i,%i)=%1.2f+%1.2f-%1.2f-%1.2f\n", idx_a, idx_b, result,idx_a1, idx_b1, idx_a1, idx_b2, idx_a2, idx_b1, idx_a2, idx_b2, k11, k22, k21, k12) ;
  
  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
