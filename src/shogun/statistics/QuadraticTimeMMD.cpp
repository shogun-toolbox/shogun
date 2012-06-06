/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/features/Features.h>

using namespace shogun;

CQuadraticTimeMMD::CQuadraticTimeMMD() : CKernelTwoSampleTestStatistic()
{
	init();
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q,
		index_t q_start) :
		CKernelTwoSampleTestStatistic(kernel, p_and_q, q_start)
{
	init();

	if (q_start!=p_and_q->get_num_vectors()/2)
	{
		SG_ERROR("CQuadraticTimeMMD: Only features with equal number of vectors "
				"are currently possible\n");
	}
}

CQuadraticTimeMMD::~CQuadraticTimeMMD()
{

}

void CQuadraticTimeMMD::init()
{
	/* TODO register parameters*/
}

float64_t CQuadraticTimeMMD::compute_statistic()
{
	/* split computations into three terms from JLMR paper (see documentation )*/
	index_t m=m_q_start;
	index_t n=m_p_and_q->get_num_vectors();

	/* init kernel with features */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* first term */
	float64_t first=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
		{
			/* ensure i!=j */
			if (i==j)
				continue;

			first+=m_kernel->kernel(i,j);
		}
	}
	first/=m*(m-1);

	/* second term */
	float64_t second=0;
	for (index_t i=m_q_start; i<n; ++i)
	{
		for (index_t j=m_q_start; j<n; ++j)
		{
			/* ensure i!=j */
			if (i==j)
				continue;

			second+=m_kernel->kernel(i,j);
		}
	}
	second/=n*(n-1);

	/* third term */
	float64_t third=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=m_q_start; j<n; ++j)
			third+=m_kernel->kernel(i,j);
	}
	third*=-2.0/(m*n);

	return first+second-third;
}

float64_t CQuadraticTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_threshold_method)
	{
		/* TODO implement new null distribution approximations here */
		default:
			result=CKernelTwoSampleTestStatistic::compute_p_value(statistic);
			break;
	}

	return result;
}

SGVector<float64_t> CQuadraticTimeMMD::sample_null_spectrum(index_t num_samples,
		index_t num_eigenvalues)
{
	/* the whole procedure is already checked against MATLAB implementation */

	if (m_q_start!=m_p_and_q->get_num_vectors()/2)
	{
		/* TODO support different numbers of samples */
		SG_ERROR("%s::sample_null_spectrum(): Currently, only equal "
				"sample sizes are supported\n", get_name());
	}

	if (num_eigenvalues>2*m_q_start-1)
	{
		SG_ERROR("%s::sample_null_spectrum(): Number of Eigenvalues too large\n",
				get_name());
	}

	/* 2m-1 is the maximum number of used eigenvalues */
	if (num_eigenvalues==-1)
		num_eigenvalues=2*m_q_start-1;

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	m_kernel->init(m_p_and_q, m_p_and_q);
	SGMatrix<float64_t> K=m_kernel->get_kernel_matrix();

	/* center matrix K=H*K*H */
	CMath::center_matrix(K.matrix, K.num_rows, K.num_cols);

	/* compute eigenvalues and select num_eigenvalues largest ones */
	SGVector<float64_t> eigenvalues=CMath::compute_eigenvectors(K);
	SGVector<float64_t> largest_ev(num_eigenvalues);

	/* scale by 1/2/m on the fly and take abs value*/
	for (index_t i=0; i<num_eigenvalues; ++i)
		largest_ev[i]=CMath::abs(1.0/2/m_q_start*eigenvalues[eigenvalues.vlen-1-i]);

	/* finally, sample from null distribution */
	SGVector<float64_t> null_samples(num_samples);
	for (index_t i=0; i<num_samples; ++i)
	{
		/* 2*sum(kEigs.*(randn(length(kEigs),1)).^2); */
		null_samples[i]=0;
		for (index_t j=0; j<largest_ev.vlen; ++j)
			null_samples[i]+=largest_ev[j]*CMath::pow(2.0, 2);

		null_samples[i]*=2;
	}

	return null_samples;
}
