/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/CharPolyKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CCharPolyKernel::CCharPolyKernel(LONG size, INT d, bool inhom, bool use_norm)
  : CSimpleKernel<CHAR>(size),degree(d),inhomogene(inhom),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), use_normalization(use_norm)
{
}

CCharPolyKernel::~CCharPolyKernel() 
{
	cleanup();
}
  
bool CCharPolyKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CSimpleKernel<CHAR>::init(l,r,do_init);

	initialized = false ;
	INT i;

	if (use_normalization)
	{
		if (sqrtdiag_lhs != sqrtdiag_rhs)
			delete[] sqrtdiag_rhs;
		sqrtdiag_rhs=NULL ;
		delete[] sqrtdiag_lhs;
		sqrtdiag_lhs=NULL ;

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

		this->lhs=(CCharFeatures*) l;
		this->rhs=(CCharFeatures*) l;

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
			this->lhs=(CCharFeatures*) r;
			this->rhs=(CCharFeatures*) r;

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

	this->lhs=(CCharFeatures*) l;
	this->rhs=(CCharFeatures*) r;

	initialized = true ;
	return result;
}

void CCharPolyKernel::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	initialized=false;
}

bool CCharPolyKernel::load_init(FILE* src)
{
	return false;
}

bool CCharPolyKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CCharPolyKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
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

  LONG sum=0;
  {
    for (INT i=0; i<ialen; i++)
	{
      sum+=((int) avec[i])*bvec[i];
	}

  }

  if (inhomogene)
	  sum+=1;

  DREAL result=sum;

  for (INT j=1; j<degree; j++)
	  result*=sum;

  result/=sqrt_both;

  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
