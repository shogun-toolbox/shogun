/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De
 */

#include <shogun/kernel/string/SubsequenceStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CSubsequenceStringKernel::CSubsequenceStringKernel()
: CStringKernel<char>(0), m_maxlen(1), m_lambda(1.0)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CSubsequenceStringKernel::CSubsequenceStringKernel(int32_t size, int32_t maxlen,
		float64_t lambda)
: CStringKernel<char>(size), m_maxlen(maxlen), m_lambda(lambda)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CSubsequenceStringKernel::CSubsequenceStringKernel(CStringFeatures<char>* l,
		CStringFeatures<char>* r, int32_t maxlen, float64_t lambda)
: CStringKernel<char>(10), m_maxlen(maxlen), m_lambda(lambda)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
	register_params();
}

CSubsequenceStringKernel::~CSubsequenceStringKernel()
{
	cleanup();
}

bool CSubsequenceStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CSubsequenceStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CSubsequenceStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	// sanity check
	REQUIRE(lhs, "lhs feature vector is not set!\n")
	REQUIRE(rhs, "rhs feature vector is not set!\n")

	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec=dynamic_cast<CStringFeatures<char>*>(lhs)
		->get_feature_vector(idx_a, alen, free_avec);
	char* bvec=dynamic_cast<CStringFeatures<char>*>(rhs)
		->get_feature_vector(idx_b, blen, free_bvec);

	REQUIRE(avec, "Feature vector for lhs is NULL!\n");
	REQUIRE(bvec, "Feature vector for rhs is NULL!\n");

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
				Kpp=m_lambda*(Kpp+m_lambda*(avec[j]==bvec[k])
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
				K+=m_lambda*m_lambda*(avec[j]==bvec[k])
					*Kp[i][j][k];
			}
		}
	}

	// cleanup
	dynamic_cast<CStringFeatures<char>*>(lhs)->free_feature_vector(avec, idx_a,
			free_avec);
	dynamic_cast<CStringFeatures<char>*>(rhs)->free_feature_vector(bvec, idx_b,
			free_bvec);

	for (index_t i=0; i<m_maxlen+1; ++i)
	{
		for (index_t j=0; j<alen; ++j)
			SG_FREE(Kp[i][j]);
		SG_FREE(Kp[i]);
	}
	SG_FREE(Kp);

	return K;
}

void CSubsequenceStringKernel::register_params()
{
	SG_ADD(&m_maxlen, "m_maxlen", "maximum length of common subsequences", ParameterProperties::HYPER);
	SG_ADD(&m_lambda, "m_lambda", "gap penalty", ParameterProperties::HYPER);
}
