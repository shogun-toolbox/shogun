/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/LocalityImprovedStringKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"

CLocalityImprovedCharKernel::CLocalityImprovedCharKernel(LONG size, INT l, INT d1, INT d2)
  : CSimpleKernel<CHAR>(size),length(l),inner_degree(d1),outer_degree(d2),match(NULL)
{
	SG_INFO( "LIK with parms: l=%d, d1=%d, d2=%d created!\n", l, d1, d2);
}

CLocalityImprovedCharKernel::~CLocalityImprovedCharKernel() 
{
	cleanup();
}

bool CLocalityImprovedCharKernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleKernel<CHAR>::init(l,r);

	if (result)
		match=new CHAR[((CCharFeatures*) l)->get_num_features()];

	return (match!=NULL && result==true);
}
  
void CLocalityImprovedCharKernel::cleanup()
{
	delete[] match;
	match = NULL;
}

bool CLocalityImprovedCharKernel::load_init(FILE* src)
{
	return false;
}

bool CLocalityImprovedCharKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CLocalityImprovedCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  ASSERT(alen==blen);

  INT i,j,t;

  // initialize match table 1 -> match;  0 -> no match
  for (i=0; i<alen; i++)
  {
	  if (avec[i]==bvec[i])
		  match[i]=1;
	  else
		  match[i]=0;
  }


  DREAL outer_sum=0;

  for (t=0; t<alen-length; t++)
  {
	  INT sum=0;
	  for (i=0; i<length; i++)
		  sum+=(i+1)*match[t+i]+(length-i)*match[t+i+length+1];

	  //add middle element + normalize with sum_i=0^2l+1 i = (2l+1)(l+1)
	  DREAL inner_sum= ((DREAL) sum + (length+1)*match[t+length]) / ((2*length+1)*(length+1));
	  DREAL s=inner_sum;

	  for (j=1; j<inner_degree; j++)
		  inner_sum*=s;

	  outer_sum+=inner_sum;
  }

  double result=outer_sum;

  for (i=1; i<outer_degree; i++)
	  result*=outer_sum;

  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) result;
}
