/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Soumyajit De
 */

#include <shogun/kernel/string/StringSubsequenceKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CStringSubsequenceKernel::CStringSubsequenceKernel()
: CStringKernel<char>(0), m_maxlen(1), m_lambda(1.0)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CStringSubsequenceKernel::CStringSubsequenceKernel(int32_t size, int32_t maxlen,
		float64_t lambda)
: CStringKernel<char>(size), m_maxlen(maxlen), m_lambda(lambda)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CStringSubsequenceKernel::CStringSubsequenceKernel(CStringFeatures<char>* l,
		CStringFeatures<char>* r, int32_t maxlen, float64_t lambda)
: CStringKernel<char>(10), m_maxlen(maxlen), m_lambda(lambda)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
	register_params();
}

CStringSubsequenceKernel::~CStringSubsequenceKernel()
{
	cleanup();
}

bool CStringSubsequenceKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CStringSubsequenceKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CStringSubsequenceKernel::compute(int32_t idx_a, int32_t idx_b)
{
	// sanity check
	REQUIRE(lhs, "lhs feature vector is not set!\n")
	REQUIRE(rhs, "rhs feature vector is not set!\n")

	SGVector<char> avec=dynamic_cast<CStringFeatures<char>*>(lhs)
		->get_feature_vector(idx_a);
	SGVector<char> bvec=dynamic_cast<CStringFeatures<char>*>(rhs)
		->get_feature_vector(idx_b);

	REQUIRE(avec.vector, "Feature vector for lhs is NULL!\n");
	REQUIRE(bvec.vector, "Feature vector for rhs is NULL!\n");

	int32_t alen=avec.size(), blen=bvec.size();

	// allocating memory for computing K' (Kp)
	float64_t ***Kp=SG_MALLOC(float64_t**, m_maxlen+1);
	for (index_t i=0; i<m_maxlen+1; ++i)
	{
		Kp[i]=SG_MALLOC(float64_t*, alen);
		for (index_t j=0; j<alen; ++j)
			Kp[i][j]=SG_CALLOC(float64_t, blen);
	}

	// initialize for 0 subsequence length for both the strings
	for (index_t j=0; j<alen; j++)
		for (index_t k=0; k<blen; ++k)
			Kp[0][j][k]=1.0;

	// computing of the K' (Kp) function using equations
	// shown in Lodhi et. al. See the class documentation for
	// definitions of Kp and Kpp
	for (index_t i=0; i<m_maxlen; i++)
	{
		for (index_t j=0; j<alen-1; j++)
		{
			float64_t Kpp=0.0;
			for (index_t k=0; k<blen-1; k++)
			{
				Kpp=m_lambda*(Kpp+m_lambda*(avec.vector[j]==bvec.vector[k])
						*Kp[i][j][k]);
				Kp[i+1][j+1][k+1]=m_lambda*Kp[i+1][j][k+1]+Kpp;
			}
		}
	}

	// compute the kernel function
	float64_t K=0.0;
	for (index_t i=0; i<m_maxlen; i++)
	{
		for (index_t j=0; j<alen; j++)
		{
			for (index_t k=0; k<blen; k++)
			{
				K+=m_lambda*m_lambda*(avec.vector[j]==bvec.vector[k])
					*Kp[i][j][k];
			}
		}
	}

	// cleanup
	for (index_t i=0; i<m_maxlen+1; ++i)
	{
		for (index_t j=0; j<alen; ++j)
			SG_FREE(Kp[i][j]);
		SG_FREE(Kp[i]);
	}
	SG_FREE(Kp);

	return K;
}

void CStringSubsequenceKernel::register_params()
{
	SG_ADD(&m_maxlen, "m_maxlen", "maximum length of common subsequences", MS_AVAILABLE);
	SG_ADD(&m_lambda, "m_lambda", "gap penalty", MS_AVAILABLE);
}
