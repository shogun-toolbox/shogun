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
#include "kernel/LinearStringKernel.h"
#include "features/StringFeatures.h"

CLinearStringKernel::CLinearStringKernel()
: CStringKernel<CHAR>(0), normal(NULL)
{
}

CLinearStringKernel::CLinearStringKernel(
	CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r)
: CStringKernel<CHAR>(0), normal(NULL)
{
	init(l, r);
}

CLinearStringKernel::~CLinearStringKernel()
{
	cleanup();
}

bool CLinearStringKernel::init(CFeatures *l, CFeatures *r)
{
	CStringKernel<CHAR>::init(l, r);
	return init_normalizer();
}

void CLinearStringKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
}

bool CLinearStringKernel::load_init(FILE *src)
{
	return false;
}

bool CLinearStringKernel::save_init(FILE *dest)
{
	return false;
}

void CLinearStringKernel::clear_normal()
{
	memset(normal, 0, lhs->get_num_vectors()*sizeof(DREAL));
}

void CLinearStringKernel::add_to_normal(INT idx, DREAL weight)
{
	INT vlen;
	CHAR* vec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, vlen);

	for (INT i=0; i<vlen; i++)
		normal[i] += weight*normalizer->normalize_lhs(vec[i], idx);
}

DREAL CLinearStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR *avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR *bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);
	return CMath::dot(avec, bvec, alen);
}

bool CLinearStringKernel::init_optimization(INT num_suppvec, INT *sv_idx,
		DREAL *alphas)
{
	INT i, alen;

	int num_feat = ((CStringFeatures<CHAR>*) lhs)->get_max_vector_length();
	ASSERT(num_feat);

	normal = new DREAL[num_feat];
	ASSERT(normal);
	clear_normal();

	for (i = 0; i<num_suppvec; i++)
	{
		CHAR *avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(sv_idx[i], alen);
		ASSERT(avec);

		for (INT j = 0; j<num_feat; j++)
			normal[j] += alphas[i]*normalizer->normalize_lhs(((double) avec[j]), sv_idx[i]);
	}
	set_is_initialized(true);
	return true;
}

bool CLinearStringKernel::delete_optimization()
{
	delete[] normal;
	normal = NULL;
	set_is_initialized(false);
	return true;
}

DREAL CLinearStringKernel::compute_optimized(INT idx_b)
{
	INT blen;
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	return normalizer->normalize_rhs(CMath::dot(normal, bvec, blen), idx_b);
}
