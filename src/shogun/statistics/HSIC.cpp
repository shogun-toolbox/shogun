/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/statistics/HSIC.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>

using namespace shogun;

CHSIC::CHSIC() : CKernelIndependenceTest()
{
	init();
}

CHSIC::CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p,
		CFeatures* q) :
		CKernelIndependenceTest(kernel_p, kernel_q, p, q)
{
	init();

	if (p && q && p->get_num_vectors()!=q->get_num_vectors())
	{
		SG_ERROR("Only features with equal number of vectors are currently "
				"possible\n");
	}
	else
		m_num_features=p->get_num_vectors();
}

CHSIC::~CHSIC()
{
}

void CHSIC::init()
{
	SG_ADD(&m_num_features, "num_features",
			"Number of features from each of the distributions",
			MS_NOT_AVAILABLE);

	m_num_features=0;
}

float64_t CHSIC::compute_statistic()
{
	SG_DEBUG("entering!\n");

	REQUIRE(m_kernel_p && m_kernel_q, "No or only one kernel specified!\n");

	REQUIRE(m_p && m_q, "features needed!\n")

	/* compute kernel matrices */
	SGMatrix<float64_t> K=get_kernel_matrix_K();
	SGMatrix<float64_t> L=get_kernel_matrix_L();

	/* center matrices (MATLAB: Kc=H*K*H) */
	K.center();

	/* compute MATLAB: sum(sum(Kc' .* (L))), which is biased HSIC */
	index_t m=m_num_features;
	SG_DEBUG("Number of samples %d!\n", m);

	float64_t result=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
			result+=K(j, i)*L(i, j);
	}

	/* return m times statistic */
	result/=m;

	SG_DEBUG("leaving!\n");

	return result;
}

float64_t CHSIC::compute_p_value(float64_t statistic)
{
	float64_t result=0;
	switch (m_null_approximation_method)
	{
	case HSIC_GAMMA:
	{
		/* fit gamma and return cdf at statistic */
		SGVector<float64_t> params=fit_null_gamma();
		result=CStatistics::gamma_cdf(statistic, params[0], params[1]);
		break;
	}

	default:
		/* sampling null is handled there */
		result=CIndependenceTest::compute_p_value(statistic);
		break;
	}

	return result;
}

float64_t CHSIC::compute_threshold(float64_t alpha)
{
	float64_t result=0;
	switch (m_null_approximation_method)
	{
	case HSIC_GAMMA:
	{
		/* fit gamma and return inverse cdf at statistic */
		SGVector<float64_t> params=fit_null_gamma();
		result=CStatistics::gamma_inverse_cdf(alpha, params[0], params[1]);
		break;
	}

	default:
		/* sampling null is handled there */
		result=CIndependenceTest::compute_threshold(alpha);
		break;
	}

	return result;
}

SGVector<float64_t> CHSIC::fit_null_gamma()
{
	REQUIRE(m_kernel_p && m_kernel_q, "No or only one kernel specified!\n");

	REQUIRE(m_p && m_q, "features needed!\n")

	index_t m=m_num_features;

	/* compute kernel matrices */
	SGMatrix<float64_t> K=get_kernel_matrix_K();
	SGMatrix<float64_t> L=get_kernel_matrix_L();

	/* compute sum and trace of uncentered kernel matrices, needed later */
	float64_t trace_K=0;
	float64_t trace_L=0;
	float64_t sum_K=0;
	float64_t sum_L=0;
	for (index_t i=0; i<m; ++i)
	{
		trace_K+=K(i,i);
		trace_L+=L(i,i);
		for (index_t j=0; j<m; ++j)
		{
			sum_K+=K(i,j);
			sum_L+=L(i,j);
		}
	}
	SG_DEBUG("sum_K: %f, sum_L: %f, trace_K: %f, trace_L: %f\n", sum_K, sum_L,
			trace_K, trace_L);

	/* center both matrices: K=H*K*H, L=H*L*H in MATLAB */
	K.center();
	L.center();

	/* compute the trace of MATLAB: (1/6 * Kc.*Lc).^2 Ü */
	float64_t trace=0;
	for (index_t i=0; i<m; ++i)
		trace+=CMath::pow(K(i,i)*L(i,i), 2);

	trace/=36.0;
	SG_DEBUG("trace %f\n", trace)

	/* compute sum of elements of MATLAB: (1/6 * Kc.*Lc).^2 */
	float64_t sum=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
			sum+=CMath::pow(K(i,j)*L(i,j), 2);
	}
	sum/=36.0;
	SG_DEBUG("sum %f\n", sum)

	/* compute MATLAB: 1/m/(m-1)*(sum(sum(varHSIC)) - sum(diag(varHSIC))),
	 * second term is bias correction */
	float64_t var_hsic=1.0/m/(m-1)*(sum-trace);
	SG_DEBUG("1.0/m/(m-1)*(sum-trace): %f\n", var_hsic)

	/* finally, compute variance of hsic under H0
	 * MATLAB: varHSIC = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3)  *  varHSIC */
	var_hsic=72.0*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3)*var_hsic;
	SG_DEBUG("var_hsic: %f\n", var_hsic)

	/* compute mean of matrices with diagonal elements zero on the base of sums
	 * and trace from K and L which were computed above */
	float64_t mu_x=1.0/m/(m-1)*(sum_K-trace_K);
	float64_t mu_y=1.0/m/(m-1)*(sum_L-trace_L);
	SG_DEBUG("mu_x: %f, mu_y: %f\n", mu_x, mu_y)

	/* compute mean under H0, MATLAB: 1/m * ( 1 +muX*muY  - muX - muY ) */
	float64_t m_hsic=1.0/m*(1+mu_x*mu_y-mu_x-mu_y);
	SG_DEBUG("m_hsic: %f\n", m_hsic)

	/* finally, compute parameters of gamma distirbution */
	float64_t a=CMath::pow(m_hsic, 2)/var_hsic;
	float64_t b=var_hsic*m/m_hsic;
	SG_DEBUG("a: %f, b: %f\n", a, b)

	SGVector<float64_t> result(2);
	result[0]=a;
	result[1]=b;

	SG_DEBUG("leaving!\n")
	return result;
}

SGVector<float64_t> CHSIC::sample_null()
{
	SG_DEBUG("entering!\n")

	/* replace current kernel via precomputed custom kernel and call superclass
	 * method */

	/* backup references to old kernels */
	CKernel* kernel_p=m_kernel_p;
	CKernel* kernel_q=m_kernel_q;

	/* init kernels before to be sure that everything is fine
	 * kernel function between two samples from different distributions
	 * is never computed - in fact, they may as well have different features */
	m_kernel_p->init(m_p, m_p);
	m_kernel_q->init(m_q, m_q);

	/* precompute kernel matrices */
	CCustomKernel* precomputed_p=new CCustomKernel(m_kernel_p);
	CCustomKernel* precomputed_q=new CCustomKernel(m_kernel_q);
	SG_REF(precomputed_p);
	SG_REF(precomputed_q);

	/* temporarily replace own kernels */
	m_kernel_p=precomputed_p;
	m_kernel_q=precomputed_q;

	/* use superclass sample_null which shuffles the entries for one
	 * distribution using index permutation on rows and columns of
	 * kernel matrix from one distribution, while accessing the other
	 * in its original order and then compute statistic */
	SGVector<float64_t> null_samples=CKernelIndependenceTest::sample_null();

	/* restore kernels */
	m_kernel_p=kernel_p;
	m_kernel_q=kernel_q;

	SG_UNREF(precomputed_p);
	SG_UNREF(precomputed_q);

	SG_DEBUG("leaving!\n")
	return null_samples;
}

void CHSIC::set_p(CFeatures* p)
{
	CIndependenceTest::set_p(p);
	m_num_features=p->get_num_vectors();
}

void CHSIC::set_q(CFeatures* q)
{
	CIndependenceTest::set_q(q);
	m_num_features=q->get_num_vectors();
}

