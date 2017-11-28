/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/kernel/string/LinearStringKernel.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CLinearStringKernel::CLinearStringKernel()
: CStringKernel<char>(0)
{
}

CLinearStringKernel::CLinearStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r)
: CStringKernel<char>(0)
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

void CLinearStringKernel::clear_normal()
{
	memset(m_normal.vector, 0, lhs->get_num_vectors()*sizeof(float64_t));
}

void CLinearStringKernel::add_to_normal(int32_t idx, float64_t weight)
{
	int32_t vlen;
	bool vfree;
	char* vec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int32_t i=0; i<vlen; i++)
		m_normal.vector[i] += weight*normalizer->normalize_lhs(vec[i], idx);

	((CStringFeatures<char>*) lhs)->free_feature_vector(vec, idx, vfree);
}

float64_t CLinearStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);
	ASSERT(alen==blen)
	SGVector<char> a_wrap(avec, alen, false);
	SGVector<char> b_wrap(bvec, blen, false);
	float64_t result = linalg::dot(a_wrap, b_wrap);
	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

bool CLinearStringKernel::init_optimization(
	int32_t num_suppvec, int32_t *sv_idx, float64_t *alphas)
{
	int32_t num_feat = ((CStringFeatures<char>*) lhs)->get_max_vector_length();
	ASSERT(num_feat)

	m_normal = SGVector<float64_t>(num_feat);
	ASSERT(m_normal.vector)
	clear_normal();

	for (int32_t i = 0; i<num_suppvec; i++)
	{
		int32_t alen;
		bool free_avec;
		char *avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(sv_idx[i], alen, free_avec);
		ASSERT(avec)

		for (int32_t j = 0; j<num_feat; j++)
		{
			m_normal.vector[j] += alphas[i]*
				normalizer->normalize_lhs(((float64_t) avec[j]), sv_idx[i]);
		}
		((CStringFeatures<char>*) lhs)->free_feature_vector(avec, sv_idx[i], free_avec);
	}
	set_is_initialized(true);
	return true;
}

bool CLinearStringKernel::delete_optimization()
{
	m_normal = SGVector<float64_t>();
	set_is_initialized(false);
	return true;
}

float64_t CLinearStringKernel::compute_optimized(int32_t idx_b)
{
	int32_t blen;
	bool free_bvec;
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);
	float64_t dot = 0.0;
	for (auto i = 0; m_normal.vlen; ++i)
		dot += m_normal[i]*(float64_t)bvec[i];
	float64_t result=normalizer->normalize_rhs(dot, idx_b);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}
