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
#include "lib/io.h"
#include "kernel/SparsePolyKernel.h"
#include "features/SparseFeatures.h"

CSparsePolyKernel::CSparsePolyKernel(INT size, INT d, bool i, bool un)
	: CSparseKernel<DREAL>(size), degree(d), inhomogene(i),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	use_normalization(un)
{
}

CSparsePolyKernel::CSparsePolyKernel(
	CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r,
	INT size, INT d, bool i, bool un)
	: CSparseKernel<DREAL>(size),degree(d),inhomogene(i), sqrtdiag_lhs(NULL),
	sqrtdiag_rhs(NULL), initialized(false), use_normalization(un)
{
	init(l,r);
}

CSparsePolyKernel::~CSparsePolyKernel() 
{
	cleanup();
}

bool CSparsePolyKernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CSparseKernel<DREAL>::init(l,r);

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

		this->lhs=(CSparseFeatures<DREAL>*) l;
		this->rhs=(CSparseFeatures<DREAL>*) l;

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
			this->lhs=(CSparseFeatures<DREAL>*) r;
			this->rhs=(CSparseFeatures<DREAL>*) r;

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

	this->lhs=(CSparseFeatures<DREAL>*) l;
	this->rhs=(CSparseFeatures<DREAL>*) r;

	initialized = true;
	return result;
}
  
void CSparsePolyKernel::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	initialized=false;
}

bool CSparsePolyKernel::load_init(FILE* src)
{
	return false;
}

bool CSparsePolyKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CSparsePolyKernel::compute(INT idx_a, INT idx_b)
{
  INT alen=0;
  INT blen=0;
  bool afree=false;
  bool bfree=false;

  TSparseEntry<DREAL>* avec=((CSparseFeatures<DREAL>*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<DREAL>* bvec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);

  DREAL sqrt_a= 1.0;
  DREAL sqrt_b= 1.0;
  if (initialized && use_normalization)
  {
	  sqrt_a=sqrtdiag_lhs[idx_a] ;
	  sqrt_b=sqrtdiag_rhs[idx_b] ;
  }

  DREAL sqrt_both=sqrt_a*sqrt_b;
  
  DREAL result=((CSparseFeatures<DREAL>*) lhs)->sparse_dot(1.0,avec, alen, bvec, blen);

  if (inhomogene)
	  result+=1;

  DREAL re=result;

  for (INT j=1; j<degree; j++)
	  result*=re;

  ((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseFeatures<DREAL>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result/sqrt_both;
}
