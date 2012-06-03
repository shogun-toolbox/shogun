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

SGVector<float64_t> CQuadraticTimeMMD::sample_null_spectrum(index_t num_samples)
{
	SGVector<float64_t> null_samples(num_samples);

	/* compute kernel matrix for XX, YY, XY */

	/* imaginary matrix Kz=[K KL; KL' L] (MATLAB notation) */

	/* centering matrix */

	/* centering of matrix Kz=H*Kz*H */

	/* compute eigenvalues */
	/*kEigs = eigs(Kz,params.numEigs); %note: this retains only largest magnitude eigenvalues
	        %empirical eigenvalues scaled by 1/2/m: see p. 2 Shawe-Tayor
	        %et al. (2005)
	kEigs = 1/2/m * abs(kEigs);
	numEigs = length(kEigs);  */

	/* finally, sample from null distribution */
	for (index_t i=0; i<num_samples; ++i)
	{
		/* 2*sum(kEigs.*(randn(length(kEigs),1)).^2); */
		null_samples[i]=0;
	}

	return null_samples;
}
