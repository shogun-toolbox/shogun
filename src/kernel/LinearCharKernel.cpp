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
#include "lib/Mathematics.h"
#include "kernel/LinearCharKernel.h"
#include "features/CharFeatures.h"

CLinearCharKernel::CLinearCharKernel(LONG size)
  : CSimpleKernel<CHAR>(size),scale(1.0),normal(NULL)
{
}

CLinearCharKernel::~CLinearCharKernel() 
{
	cleanup();
}
  
bool CLinearCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSimpleKernel<CHAR>::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message(M_INFO, "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CLinearCharKernel::init_rescale()
{
	LONGREAL sum=0;
	scale=1.0;
	for (LONG i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	if ( sum > (pow((double) 2, (double) 8*sizeof(LONG))) ) {
      sg_error(sg_err_fun,"the sum %lf does not fit into integer of %d bits expect bogus results.\n", sum, 8*sizeof(LONG));
   }
	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CLinearCharKernel::cleanup()
{
	delete_optimization();
}

bool CLinearCharKernel::load_init(FILE* src)
{
	return false;
}

bool CLinearCharKernel::save_init(FILE* dest)
{
	return false;
}

void CLinearCharKernel::clear_normal()
{
	int num = lhs->get_num_vectors();

	for (int i=0; i<num; i++)
		normal[i]=0;
}

void CLinearCharKernel::add_to_normal(INT idx, DREAL weight) 
{
	INT vlen;
	bool vfree;
	CHAR* vec=((CCharFeatures*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int i=0; i<vlen; i++)
		normal[i]+= weight*vec[i];

	((CCharFeatures*) lhs)->free_feature_vector(vec, idx, vfree);
}
  
DREAL CLinearCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  ASSERT(alen==blen);
  double sum=0;
  for (INT i=0; i<alen; i++)
	  sum+=((LONG) avec[i])*((LONG) bvec[i]);

  DREAL result=sum/scale;
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CLinearCharKernel::init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas) 
{
	CIO::message(M_DEBUG,"drin gelandet yeah\n");
	INT alen;
	bool afree;
	INT i;

	int num_feat=((CCharFeatures*) lhs)->get_num_features();
	ASSERT(num_feat);

	normal=new DREAL[num_feat];
	ASSERT(normal);

	for (i=0; i<num_feat; i++)
		normal[i]=0;

	for (i=0; i<num_suppvec; i++)
	{
		CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(sv_idx[i], alen, afree);
		ASSERT(avec);

		for (int j=0; j<num_feat; j++)
			normal[j]+= alphas[i] * ((double) avec[j]);

		((CCharFeatures*) lhs)->free_feature_vector(avec, 0, afree);
	}

	set_is_initialized(true);
	return true;
}

bool CLinearCharKernel::delete_optimization()
{
	delete[] normal;
	normal=NULL;
	set_is_initialized(false);
	return true;
}

DREAL CLinearCharKernel::compute_optimized(INT idx_b) 
{
	INT blen;
	bool bfree;

	CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	double result=0;
	{
		for (INT i=0; i<blen; i++)
			result+= normal[i] * ((double) bvec[i]);
	}
	result/=scale;

	((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
