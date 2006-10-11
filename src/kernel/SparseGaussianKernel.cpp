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
#include "lib/io.h"
#include "kernel/SparseGaussianKernel.h"
#include "features/Features.h"
#include "features/SparseRealFeatures.h"
#include "features/SparseFeatures.h"

CSparseGaussianKernel::CSparseGaussianKernel(INT size, double w)
  : CSparseKernel<DREAL>(size),width(w),sq_lhs(NULL),sq_rhs(NULL)
{
}

CSparseGaussianKernel::~CSparseGaussianKernel() 
{
	cleanup();
}
  
bool CSparseGaussianKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	INT len=0;;
	bool do_free=false;

	///free sq_{r,l}hs first
	cleanup();

	CSparseKernel<DREAL>::init(l, r, do_init); 

	sq_lhs= new DREAL[lhs->get_num_vectors()];
	ASSERT(sq_lhs);


	for (INT i=0; i<lhs->get_num_vectors(); i++)
	{
		sq_lhs[i]=0;
		TSparseEntry<DREAL>* vec = ((CSparseRealFeatures*) lhs)->get_sparse_feature_vector(i, len, do_free);

		for (INT j=0; j<len; j++)
			sq_lhs[i] += vec[j].entry * vec[j].entry;

		((CSparseRealFeatures*) lhs)->free_feature_vector(vec, i, do_free);
	}

	if (lhs==rhs)
		sq_rhs=sq_lhs;
	else
	{
		sq_rhs= new DREAL[rhs->get_num_vectors()];
		ASSERT(sq_rhs);

		for (INT i=0; i<rhs->get_num_vectors(); i++)
		{
			sq_rhs[i]=0;
			TSparseEntry<DREAL>* vec = ((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(i, len, do_free);

			for (INT j=0; j<len; j++)
				sq_rhs[i] += vec[j].entry * vec[j].entry;

			((CSparseRealFeatures*) rhs)->free_feature_vector(vec, i, do_free);
		}
	}
	
	return true;
}

void CSparseGaussianKernel::cleanup()
{
	if (sq_lhs != sq_rhs)
		delete[] sq_rhs;
	sq_rhs = NULL;

	delete[] sq_lhs;
	sq_lhs = NULL;
}

bool CSparseGaussianKernel::load_init(FILE* src)
{
	return false;
}

bool CSparseGaussianKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CSparseGaussianKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  TSparseEntry<DREAL>* avec=((CSparseRealFeatures*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<DREAL>* bvec=((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
  
  DREAL result = sq_lhs[idx_a] + sq_rhs[idx_b];

  if (alen<=blen)
  {
	  INT j=0;
	  for (INT i=0; i<alen; i++)
	  {
		  INT a_feat_idx=avec[i].feat_index;

		  while ( (j<blen) && (bvec[j].feat_index < a_feat_idx) )
			  j++;

		  if ( (j<blen) && (bvec[j].feat_index == a_feat_idx) )
		  {
			  result-= 2*(avec[i].entry*bvec[j].entry);
			  j++;
		  }
	  }
  }
  else
  {
	  INT j=0;
	  for (INT i=0; i<blen; i++)
	  {
		  INT b_feat_idx=bvec[i].feat_index;

		  while ( (j<alen) && (avec[j].feat_index < b_feat_idx) )
			  j++;

		  if ( (j<alen) && (avec[j].feat_index == b_feat_idx) )
		  {
			  result-= 2*(bvec[i].entry*avec[j].entry);
			  j++;
		  }
	  }
  }
  result=exp(-result/width);

  ((CSparseRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
