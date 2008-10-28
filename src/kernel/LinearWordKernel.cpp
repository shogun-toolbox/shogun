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
#include "kernel/LinearWordKernel.h"
#include "features/WordFeatures.h"

CLinearWordKernel::CLinearWordKernel()
: CSimpleKernel<uint16_t>(0), normal(NULL)
{
}

CLinearWordKernel::CLinearWordKernel(CWordFeatures* l, CWordFeatures* r)
: CSimpleKernel<uint16_t>(0), normal(NULL)
{
	init(l, r);
}

CLinearWordKernel::~CLinearWordKernel()
{
	cleanup();
}

bool CLinearWordKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<uint16_t>::init(l, r);
	return init_normalizer();
}

void CLinearWordKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
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
	int32_t num = lhs->get_num_vectors();
	CMath::fill_vector(normal, num, 0.0);
}

void CLinearWordKernel::add_to_normal(int32_t idx, float64_t weight)
{
	int32_t vlen;
	bool vfree;
	uint16_t* vec=((CWordFeatures*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int32_t i=0; i<vlen; i++)
		normal[i]+= weight*normalizer->normalize_lhs(vec[i], idx);

	((CWordFeatures*) lhs)->free_feature_vector(vec, idx, vfree);
}

float64_t CLinearWordKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	uint16_t* avec=((CWordFeatures*) lhs)->get_feature_vector(
		idx_a, alen, afree);
	uint16_t* bvec=((CWordFeatures*) rhs)->get_feature_vector(
		idx_b, blen, bfree);
	ASSERT(alen==blen);

	float64_t result=CMath::dot(avec, bvec, alen);

	((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

bool CLinearWordKernel::init_optimization(
	int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas)
{
	int32_t alen;
	bool afree;

	int32_t num_feat=((CWordFeatures*) lhs)->get_num_features();
	ASSERT(num_feat);

	normal=new float64_t[num_feat];
	CMath::fill_vector(normal, num_feat, 0.0);

	for (int32_t i=0; i<num_suppvec; i++)
	{
		uint16_t* avec=((CWordFeatures*) lhs)->get_feature_vector(
			sv_idx[i], alen, afree);
		ASSERT(avec);

		for (int32_t j=0; j<num_feat; j++)
			normal[j]+=alphas[i] * normalizer->normalize_lhs(
				((float64_t) avec[j]), sv_idx[i]);

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

float64_t CLinearWordKernel::compute_optimized(int32_t idx_b) 
{
	int32_t blen;
	bool bfree;

	uint16_t* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	float64_t result=0;
	{
		for (int32_t i=0; i<blen; i++)
			result+= normal[i] * ((float64_t) bvec[i]);
	}

	((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return normalizer->normalize_rhs(result, idx_b);
}
