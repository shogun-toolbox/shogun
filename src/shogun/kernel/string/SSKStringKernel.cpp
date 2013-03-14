/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 * Copyright (C) 2013 Indian Institute of Technology, Bombay
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/SSKStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CSSKStringKernel::CSSKStringKernel()
: CStringKernel<char>(0), m_maxlen(DEFAULT_MAXLEN), m_lambda(DEFAULT_LAMBDA)
{
	init();
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CSSKStringKernel::CSSKStringKernel(int32_t size, int32_t maxlen, float64_t lambda)
: CStringKernel<char>(size), m_maxlen(maxlen), m_lambda(lambda)
{
	init();
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CSSKStringKernel::CSSKStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r,
	int32_t maxlen, float64_t lambda)
: CStringKernel<char>(10), m_maxlen(maxlen), m_lambda(lambda)
{
	init();
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
	register_params();
}

CSSKStringKernel::~CSSKStringKernel()
{
	cleanup();
}

bool CSSKStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CSSKStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CSSKStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	float64_t ***Kp = new float64_t**[m_maxlen + 1];

	for( int i = 0; i < m_maxlen + 1; ++i ) {
		Kp[i] = new float64_t*[alen];
		for( int j = 0; j < alen; ++j ) {
			Kp[i][j] = new float64_t[blen];
		}
	}

	for( int j = 0; j < alen; j++ )
		for( int k = 0; k < blen; k++ )
			Kp[0][j][k] = 1;

	for (int i = 0; i < m_maxlen; i++) {
		for (int j = 0; j < alen - 1; j++) {
			float64_t Kpp = 0.0;
			for (int k = 0; k < blen - 1; k++) {
				Kpp = m_lambda * (Kpp + m_lambda * (avec[j] == bvec[k])  * Kp[i][j][k]);
				Kp[i + 1][j + 1][k + 1] = m_lambda * Kp[i + 1][j][k + 1] + Kpp;
			}
		}
	}

	float64_t *K = new float64_t[m_maxlen];
	for (int l = 0; l < m_maxlen; l++) {
		K[l] = 0.0;
		for (int j = 0; j < alen; j++) {
			for (int k = 0; k < blen; k++)
				K[l] += m_lambda * m_lambda * (avec[j] == bvec[k]) * Kp[l][j][k];
		}
	}

	float64_t total = 0;
	for (int i = 0; i < m_maxlen; i++)
		total += K[i];

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	SG_FREE(Kp);
	SG_FREE(K);

	return total;
}

void CSSKStringKernel::register_params()
{
	SG_ADD(&m_maxlen, "m_maxlen", "maximum length of common subsequences", MS_AVAILABLE);
	SG_ADD(&m_lambda, "m_lambda", "gap penalty", MS_AVAILABLE);
}

void CSSKStringKernel::init()
{
	// do nothing
}
