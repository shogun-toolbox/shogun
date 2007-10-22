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
#include "kernel/GaussianShiftKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CGaussianShiftKernel::CGaussianShiftKernel(INT size, double w, int p_max_shift, int p_shift_step)
	: CGaussianKernel(size, w), max_shift(p_max_shift), shift_step(p_shift_step)
{
}

CGaussianShiftKernel::CGaussianShiftKernel(CRealFeatures* l, CRealFeatures* r, double w, int p_max_shift, int p_shift_step, INT size)
	: CGaussianKernel(l, r, w, size), max_shift(p_max_shift), shift_step(p_shift_step)
{
	init(l,r);
}

CGaussianShiftKernel::~CGaussianShiftKernel()
{
}
  
DREAL CGaussianShiftKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  ASSERT(alen==blen);

  INT ialen=(int) alen;

  DREAL result = 0.0 ;
  DREAL sum=0.0 ;
  for (INT i=0; i<ialen; i++)
	  sum+=(avec[i]-bvec[i])*(avec[i]-bvec[i]);
  result += exp(-sum/width) ;
  
  for (INT shift = shift_step, s=1; shift<max_shift; shift+=shift_step, s++)
  {
	  sum=0.0 ;
	  for (INT i=0; i<ialen-shift; i++)
		  sum+=(avec[i+shift]-bvec[i])*(avec[i+shift]-bvec[i]);
	  result += exp(-sum/width)/(2*s) ;

	  sum=0.0 ;
	  for (INT i=0; i<ialen-shift; i++)
		  sum+=(avec[i]-bvec[i+shift])*(avec[i]-bvec[i+shift]);
	  result += exp(-sum/width)/(2*s) ;
  }
  
  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
