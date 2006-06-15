/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "kernel/PolyKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

CPolyKernel::CPolyKernel(LONG size, INT d, bool inhom, bool use_norm)
  : CRealKernel(size),degree(d),inhomogene(inhom),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), use_normalization(use_norm)
{
}

CPolyKernel::~CPolyKernel() 
{
	cleanup();
}
  
bool CPolyKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CRealKernel::init(l,r,do_init);

	initialized = false ;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
	  delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL ;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL ;

	if (use_normalization)
	{
		sqrtdiag_lhs= new DREAL[lhs->get_num_vectors()];

		for (i=0; i<lhs->get_num_vectors(); i++)
			sqrtdiag_lhs[i]=1;

		if (l==r)
			sqrtdiag_rhs=sqrtdiag_lhs;
		else
		{
			sqrtdiag_rhs= new DREAL[rhs->get_num_vectors()];
			for (i=0; i<rhs->get_num_vectors(); i++)
				sqrtdiag_rhs[i]=1;
		}

		ASSERT(sqrtdiag_lhs);
		ASSERT(sqrtdiag_rhs);

		this->lhs=(CRealFeatures*) l;
		this->rhs=(CRealFeatures*) l;

		//compute normalize to 1 values
		for (i=0; i<lhs->get_num_vectors(); i++)
		{
			sqrtdiag_lhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_lhs[i]==0)
				sqrtdiag_lhs[i]=1e-16;
		}

		// if lhs is different from rhs (train/test data)
		// compute also the normalization for rhs
		if (sqrtdiag_lhs!=sqrtdiag_rhs)
		{
			this->lhs=(CRealFeatures*) r;
			this->rhs=(CRealFeatures*) r;

			//compute normalize to 1 values
			for (i=0; i<rhs->get_num_vectors(); i++)
			{
				sqrtdiag_rhs[i]=sqrt(compute(i,i));

				//trap divide by zero exception
				if (sqrtdiag_rhs[i]==0)
					sqrtdiag_rhs[i]=1e-16;
			}
		}
	}

	this->lhs=(CRealFeatures*) l;
	this->rhs=(CRealFeatures*) r;

	initialized = true ;
	return result;
}

void CPolyKernel::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	initialized=false;
}

bool CPolyKernel::load_init(FILE* src)
{
	return false;
}

bool CPolyKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CPolyKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  ASSERT(alen==blen);

  DREAL sqrt_a= 1 ;
  DREAL sqrt_b= 1 ;
  if (initialized && use_normalization)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  DREAL sqrt_both=sqrt_a*sqrt_b;

  INT ialen=(int) alen;

#ifndef HAVE_LAPACK
  DREAL result=0;
  {
    for (INT i=0; i<ialen; i++)
	{
      result+=avec[i]*bvec[i];
	}

  }
#else
  INT skip=1;
  DREAL result=cblas_ddot(ialen, avec, skip, bvec, skip);
#endif

  if (inhomogene)
	  result+=1;

  DREAL re=result;

  for (INT j=1; j<degree; j++)
	  result*=re;

  result/=sqrt_both;

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
