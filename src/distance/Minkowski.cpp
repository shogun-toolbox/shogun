/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/Minkowski.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

CMinkowskiMetric::CMinkowskiMetric(DREAL p)
  : CRealDistance(),k(p)
{
}

CMinkowskiMetric::~CMinkowskiMetric() 
{
	cleanup();
}
  
bool CMinkowskiMetric::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CRealDistance::init(l,r,do_init);

	return result;
}

void CMinkowskiMetric::cleanup()
{
}

bool CMinkowskiMetric::load_init(FILE* src)
{
	return false;
}

bool CMinkowskiMetric::save_init(FILE* dest)
{
	return false;
}
  
DREAL CMinkowskiMetric::compute(INT idx_a, INT idx_b)
{
	
  INT alen, blen;
  bool afree, bfree;

  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  ASSERT(alen==blen);
  INT ialen=(int) alen;

  DREAL absTmp = 0;
  DREAL result=0;
  {
    for (INT i=0; i<ialen; i++)
	{
      	absTmp=fabs(avec[i]-bvec[i]);
	result+=pow(absTmp,k);
	}

  }

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
  
  
  return pow(result,1/k);
}

