/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <statistics/HSIC.h>
#include <features/Features.h>
#include <mathematics/Statistics.h>
#include <kernel/Kernel.h>
#include <kernel/CustomKernel.h>

using namespace shogun;

CHSIC::CHSIC() :
		CKernelIndependenceTestStatistic()
{
	init();
}

CHSIC::CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p_and_q,
		index_t m) :
		CKernelIndependenceTestStatistic(kernel_p, kernel_q, m_p_and_q, m)
{
	if (p_and_q && p_and_q->get_num_vectors()/2!=m)
	{
		SG_ERROR("%s: Only features with equal number of vectors are currently "
				"possible\n", get_name());
	}

	init();
}

CHSIC::CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p,
		CFeatures* q) :
		CKernelIndependenceTestStatistic(kernel_p, kernel_q, p, q)
{
	if (p && q && p->get_num_vectors()!=q->get_num_vectors())
	{
		SG_ERROR("%s: Only features with equal number of vectors are currently "
				"possible\n", get_name());
	}

	init();
}

CHSIC::~CHSIC()
{

}

void CHSIC::init()
{

}

float64_t CHSIC::compute_statistic()
{
	REQUIRE(m_kernel_p && m_kernel_q, "%s::fit_null_gamma(): No or only one "
			"kernel specified!\n", get_name());

	REQUIRE(m_p_and_q, "%s::compute_statistic: features needed!\n", get_name())

	/* compute kernel matrices */
	SGMatrix<float64_t> K=get_kernel_matrix_K();
	SGMatrix<float64_t> L=get_kernel_matrix_L();

	/* center matrices (MATLAB: Kc=H*K*H) */
	K.center();

	/* compute MATLAB: sum(sum(Kc' .* (L))), which is biased HSIC */
	index_t m=m_m;
	float64_t result=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
			result+=K(j, i)*L(i, j);
	}

	/* return m times statistic */
	result/=m;

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
		/* bootstrapping is handled there */
		result=CTwoDistributionsTestStatistic::compute_p_value(statistic);
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
		result=CStatistics::inverse_gamma_cdf(alpha, params[0], params[1]);
		break;
	}

	default:
		/* bootstrapping is handled there */
		result=CTwoDistributionsTestStatistic::compute_threshold(alpha);
		break;
	}

	return result;
}

SGVector<float64_t> CHSIC::fit_null_gamma()
{
	REQUIRE(m_kernel_p && m_kernel_q, "%s::fit_null_gamma(): No or only one "
			"kernel specified!\n", get_name());

	REQUIRE(m_p_and_q, "%s::fit_null_gamma: features needed!\n", get_name())

	index_t m=m_m;

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

	SG_DEBUG(("leaving %s::fit_null_gamma()\n"), get_name())
	return result;
}

SGMatrix<float64_t> CHSIC::get_kernel_matrix_K()
{
	SGMatrix<float64_t> K;

	/* subset for selecting only data from one distribution */
	SGVector<index_t> subset(m_m);
	subset.range_fill();

	/* distinguish between custom and normal kernels */
	if (m_kernel_p->get_kernel_type()==K_CUSTOM)
	{
		/* custom kernels need to to be initialised when a subset is added */
		CCustomKernel* custom_kernel_p=(CCustomKernel*)m_kernel_p;
		custom_kernel_p->add_row_subset(subset);
		custom_kernel_p->add_col_subset(subset);

		K=custom_kernel_p->get_kernel_matrix();

		custom_kernel_p->remove_row_subset();
		custom_kernel_p->remove_col_subset();
	}
	else
	{
		m_p_and_q->add_subset(subset);
		m_kernel_p->init(m_p_and_q, m_p_and_q);
		K=m_kernel_p->get_kernel_matrix();
		m_p_and_q->remove_subset();

		/* re-init kernel since subset was changed */
		m_kernel_p->init(m_p_and_q, m_p_and_q);
	}

	return K;
}

SGMatrix<float64_t> CHSIC::get_kernel_matrix_L()
{
	SGMatrix<float64_t> L;

	/* subset for selecting only data from one distribution */
	SGVector<index_t> subset(m_m);
	subset.range_fill();
	subset.add(m_m);

	/* now second half of data for L */
	if (m_kernel_q->get_kernel_type()==K_CUSTOM)
	{
		/* custom kernels need to to be initialised when a subset is added */
		CCustomKernel* custom_kernel_q=(CCustomKernel*)m_kernel_q;
		custom_kernel_q->add_row_subset(subset);
		custom_kernel_q->add_col_subset(subset);

		L=custom_kernel_q->get_kernel_matrix();

		custom_kernel_q->remove_row_subset();
		custom_kernel_q->remove_col_subset();
	}
	else
	{
		m_p_and_q->add_subset(subset);
		m_kernel_q->init(m_p_and_q, m_p_and_q);
		L=m_kernel_q->get_kernel_matrix();
		m_p_and_q->remove_subset();

		/* re-init kernel since subset was changed */
		m_kernel_q->init(m_p_and_q, m_p_and_q);
	}

	return L;
}

SGVector<float64_t> CHSIC::bootstrap_null()
{
	SG_DEBUG("entering CHSIC::bootstrap_null()\n")

	/* replace current kernel via precomputed custom kernel and call superclass
	 * method */

	/* backup references to old kernels */
	CKernel* kernel_p=m_kernel_p;
	CKernel* kernel_q=m_kernel_q;

	/* init kernels before to be sure that everything is fine */
	m_kernel_p->init(m_p_and_q, m_p_and_q);
	m_kernel_q->init(m_p_and_q, m_p_and_q);

	/* precompute kernel matrices */
	CCustomKernel* precomputed_p=new CCustomKernel(m_kernel_p);
	CCustomKernel* precomputed_q=new CCustomKernel(m_kernel_q);
	SG_REF(precomputed_p);
	SG_REF(precomputed_q);

	/* temporarily replace own kernels */
	m_kernel_p=precomputed_p;
	m_kernel_q=precomputed_q;

	/* use superclass bootstrapping which permutes custom kernels */
	SGVector<float64_t> null_samples=
			CKernelIndependenceTestStatistic::bootstrap_null();

	/* restore kernels */
	m_kernel_p=kernel_p;
	m_kernel_q=kernel_q;

	SG_UNREF(precomputed_p);
	SG_UNREF(precomputed_q);


	SG_DEBUG("leaving CHSIC::bootstrap_null()\n")
	return null_samples;
}
