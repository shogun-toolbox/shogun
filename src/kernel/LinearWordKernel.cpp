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
#include "lib/Mathematics.h"
#include "kernel/LinearWordKernel.h"
#include "features/WordFeatures.h"

CLinearWordKernel::CLinearWordKernel(LONG size, bool do_rescale_, DREAL scale_)
  : CSimpleKernel<WORD>(size),scale(scale_),do_rescale(do_rescale_),initialized(false),
	normal(NULL)
{
}

CLinearWordKernel::~CLinearWordKernel() 
{
	cleanup();
}

bool CLinearWordKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<WORD>::init(l, r);

	if (!initialized)
		init_rescale() ;

	SG_INFO( "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CLinearWordKernel::init_rescale()
{
	if (!do_rescale)
		return ;
	LONGREAL sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
		sum+=compute(i, i);

	if ( sum > (pow((double) 2, (double) 8*sizeof(LONG))) ) {
      SG_ERROR( "the sum %lf does not fit into integer of %d bits expect bogus results.\n", sum, 8*sizeof(LONG));
   }
	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
	initialized=true;
}

void CLinearWordKernel::cleanup()
{
	delete_optimization();
}

bool CLinearWordKernel::load_init(FILE* src)
{
	return false;
}

bool CLinearWordKernel::save_init(FILE* dest)
{
	return false;
}

void CLinearWordKernel::clear_normal()
{
	int num = lhs->get_num_vectors();

	for (int i=0; i<num; i++)
		normal[i]=0;
}

void CLinearWordKernel::add_to_normal(INT idx, DREAL weight) 
{
	INT vlen;
	bool vfree;
	WORD* vec=((CWordFeatures*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int i=0; i<vlen; i++)
		normal[i]+= weight*vec[i];

	((CWordFeatures*) lhs)->free_feature_vector(vec, idx, vfree);
}
  
DREAL CLinearWordKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);
	double sum=0;
	for (LONG i=0; i<alen; i++)
		sum+=((LONG) avec[i])*((LONG) bvec[i]);

	DREAL result=sum/scale;
	((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

bool CLinearWordKernel::init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas) 
{
	SG_DEBUG("drin gelandet yeah\n");
	INT alen;
	bool afree;

	int num_feat=((CWordFeatures*) lhs)->get_num_features();
	ASSERT(num_feat);

	normal=new DREAL[num_feat];
	ASSERT(normal);

	for (INT i=0; i<num_feat; i++)
		normal[i]=0;

	for (int i=0; i<num_suppvec; i++)
	{
		WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(sv_idx[i], alen, afree);
		ASSERT(avec);

		for (int j=0; j<num_feat; j++)
			normal[j]+=alphas[i] * ((double) avec[j]);

		((CWordFeatures*) lhs)->free_feature_vector(avec, 0, afree);
	}

	set_is_initialized(true);
	return true;
}

bool CLinearWordKernel::delete_optimization()
{
	delete[] normal;
	normal=NULL;
	set_is_initialized(false);

	return true;
}

DREAL CLinearWordKernel::compute_optimized(INT idx_b) 
{
	INT blen;
	bool bfree;

	WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	double result=0;
	{
		for (INT i=0; i<blen; i++)
			result+= normal[i] * ((double) bvec[i]);
	}
	result/=scale;

	((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
