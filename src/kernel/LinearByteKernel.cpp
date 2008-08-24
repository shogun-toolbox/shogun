/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "kernel/LinearByteKernel.h"
#include "features/ByteFeatures.h"

CLinearByteKernel::CLinearByteKernel()
: CSimpleKernel<BYTE>(0), normal(NULL)
{
}

CLinearByteKernel::CLinearByteKernel(CByteFeatures* l, CByteFeatures* r)
: CSimpleKernel<BYTE>(0), normal(NULL)
{
	init(l, r);
}

CLinearByteKernel::~CLinearByteKernel()
{
	cleanup();
}

bool CLinearByteKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<BYTE>::init(l, r);
	return init_normalizer();
}

void CLinearByteKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
}

bool CLinearByteKernel::load_init(FILE* src)
{
	return false;
}

bool CLinearByteKernel::save_init(FILE* dest)
{
	return false;
}

void CLinearByteKernel::clear_normal()
{
	int num = lhs->get_num_vectors();

	for (int i=0; i<num; i++)
		normal[i]=0;
}

void CLinearByteKernel::add_to_normal(INT idx, DREAL weight) 
{
	INT vlen;
	bool vfree;
	BYTE* vec=((CByteFeatures*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int i=0; i<vlen; i++)
		normal[i]+= weight*normalizer->normalize_lhs(vec[i], idx);

	((CByteFeatures*) lhs)->free_feature_vector(vec, idx, vfree);
}
  
DREAL CLinearByteKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  BYTE* avec=((CByteFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  BYTE* bvec=((CByteFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  ASSERT(alen==blen);

  DREAL result=CMath::dot(avec,bvec, alen);

  ((CByteFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CByteFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CLinearByteKernel::init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas) 
{
	INT alen;
	bool afree;

	int num_feat=((CByteFeatures*) lhs)->get_num_features();
	ASSERT(num_feat);

	normal=new DREAL[num_feat];
	for (INT i=0; i<num_feat; i++)
		normal[i]=0;

	for (int i=0; i<num_suppvec; i++)
	{
		BYTE* avec=((CByteFeatures*) lhs)->get_feature_vector(sv_idx[i], alen, afree);
		ASSERT(avec);

		for (int j=0; j<num_feat; j++)
			normal[j]+= alphas[i] * normalizer->normalize_lhs(((double) avec[j]), sv_idx[i]);

		((CByteFeatures*) lhs)->free_feature_vector(avec, 0, afree);
	}

	set_is_initialized(true);
	return true;
}

bool CLinearByteKernel::delete_optimization()
{
	delete[] normal;
	normal=NULL;

	set_is_initialized(false);

	return true;
}

DREAL CLinearByteKernel::compute_optimized(INT idx_b) 
{
	INT blen;
	bool bfree;

	BYTE* bvec=((CByteFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	double result=0;
	{
		for (INT i=0; i<blen; i++)
			result+= normal[i] * ((double) bvec[i]);
	}

	((CByteFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return normalizer->normalize_rhs(result, idx_b);
}
