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
: CStringKernel<char>(0), normal(NULL)
{
}

CLinearStringKernel::CLinearStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r)
: CStringKernel<char>(0), normal(NULL)
{
	init(l, r);
}

CLinearStringKernel::~CLinearStringKernel()
{
	cleanup();
}

bool CLinearStringKernel::init(CFeatures *l, CFeatures *r)
{
	CStringKernel<char>::init(l, r);
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
	memset(normal, 0, lhs->get_num_vectors()*sizeof(float64_t));
}

void CLinearStringKernel::add_to_normal(int32_t idx, float64_t weight)
{
	int32_t vlen;
	char* vec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx, vlen);

	for (int32_t i=0; i<vlen; i++)
		normal[i] += weight*normalizer->normalize_lhs(vec[i], idx);
}

float64_t CLinearStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;

	char *avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen);
	char *bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);
	return CMath::dot(avec, bvec, alen);
}

bool CLinearStringKernel::init_optimization(
	int32_t num_suppvec, int32_t *sv_idx, float64_t *alphas)
{
	int32_t i, alen;

	int32_t num_feat = ((CStringFeatures<char>*) lhs)->get_max_vector_length();
	ASSERT(num_feat);

	normal = new float64_t[num_feat];
	ASSERT(normal);
	clear_normal();

	for (i = 0; i<num_suppvec; i++)
	{
		char *avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(sv_idx[i], alen);
		ASSERT(avec);

		for (int32_t j = 0; j<num_feat; j++)
		{
			normal[j] += alphas[i]*
				normalizer->normalize_lhs(((float64_t) avec[j]), sv_idx[i]);
		}
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

float64_t CLinearStringKernel::compute_optimized(int32_t idx_b)
{
	int32_t blen;
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen);

	return normalizer->normalize_rhs(CMath::dot(normal, bvec, blen), idx_b);
}
