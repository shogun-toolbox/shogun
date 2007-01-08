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
#include "features/Features.h"
#include "features/SparseFeatures.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparseKernel.h"

CSparseLinearKernel::CSparseLinearKernel(INT size, bool do_rescale_, DREAL scale_)
  : CSparseKernel<DREAL>(size),scale(scale_),do_rescale(do_rescale_), normal_length(0), normal(NULL)
{
	properties |= KP_LINADD;
}

CSparseLinearKernel::~CSparseLinearKernel() 
{
	cleanup();
}
  
bool CSparseLinearKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSparseKernel<DREAL>::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message(M_INFO, "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CSparseLinearKernel::init_rescale()
{
	if (!do_rescale)
		return ;
	double sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CSparseLinearKernel::cleanup()
{
	delete_optimization();
}

bool CSparseLinearKernel::load_init(FILE* src)
{
	return false;
}

bool CSparseLinearKernel::save_init(FILE* dest)
{
	return false;
}

void CSparseLinearKernel::clear_normal()
{
	int num = ((CSparseFeatures<DREAL>*) lhs)->get_num_features();
	if (normal==NULL)
		normal = new DREAL[num] ;
	for (int i=0; i<num; i++)
		normal[i]=0;
}

void CSparseLinearKernel::add_to_normal(INT idx, DREAL weight) 
{
	INT vlen;
	bool vfree;
	TSparseEntry<DREAL>* vec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx, vlen, vfree);

	for (int i=0; i<vlen; i++)
		normal[vec[i].feat_index]+= weight*vec[i].entry;

	((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(vec, idx, vfree);
}
  
DREAL CSparseLinearKernel::compute(INT idx_a, INT idx_b)
{
  INT alen=0;
  INT blen=0;
  bool afree=false;
  bool bfree=false;

  TSparseEntry<DREAL>* avec=((CSparseFeatures<DREAL>*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<DREAL>* bvec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
  
  DREAL result=0;

  //result remains zero when one of the vectors is non existent
  if (avec && bvec)
  {
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
	    		  result+= avec[i].entry * bvec[j].entry;
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
	    		  result+= bvec[i].entry * avec[j].entry;
	    		  j++;
	    	  }
	      }
	  }

	  result/=scale;
  }

  ((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseFeatures<DREAL>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CSparseLinearKernel::init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas) 
{
	clear_normal();

	for (int i=0; i<num_suppvec; i++)
		add_to_normal(sv_idx[i], alphas[i]);

	set_is_initialized(true);
	return true;;
}

bool CSparseLinearKernel::delete_optimization()
{
	delete[] normal;
	normal=NULL;
	set_is_initialized(false);

	return true;
}

DREAL CSparseLinearKernel::compute_optimized(INT idx) 
{
	INT vlen;
	bool vfree;

	TSparseEntry<DREAL>* vec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx, vlen, vfree);

	DREAL result=0;
	for (INT i=0; i<vlen; i++)
		result+=normal[vec[i].feat_index]*vec[i].entry;
	result/=scale;

	((CSparseFeatures<DREAL>*) rhs)->free_sparse_feature_vector(vec, idx, vfree);

	return result;
}
