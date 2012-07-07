/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

CLinearTimeMMD::CLinearTimeMMD() :
		CKernelTwoSampleTestStatistic()
{
	init();
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q,
		index_t q_start) :
		CKernelTwoSampleTestStatistic(kernel, p_and_q, q_start)
{
	init();

	if (p_and_q && q_start!=p_and_q->get_num_vectors()/2)
	{
		SG_ERROR("CLinearTimeMMD: Only features with equal number of vectors "
				"are currently possible\n");
	}
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CFeatures* p, CFeatures* q) :
		CKernelTwoSampleTestStatistic(kernel, p, q)
{
	init();

	if (p->get_num_vectors()!=q->get_num_vectors())
	{
		SG_ERROR("CLinearTimeMMD: Only features with equal number of vectors "
				"are currently possible\n");
	}
}

CLinearTimeMMD::~CLinearTimeMMD()
{

}

void CLinearTimeMMD::init()
{

}

float64_t CLinearTimeMMD::compute_statistic()
{
	SG_DEBUG("entering CLinearTimeMMD::compute_statistic()\n");

	if (!m_kernel)
		SG_ERROR("%s::compute_statistic(): No kernel specified!\n", get_name());

	/* TODO features with a different number of vectors should be allowed */

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m=m_q_start;
	index_t m_2=m/2;

	SG_DEBUG("m_q_start=%d\n", m_q_start);

	/* compute traces of kernel matrices for linear MMD */
	m_kernel->init(m_p_and_q, m_p_and_q);

	float64_t pp=0;
	float64_t qq=0;
	float64_t pq=0;
	float64_t qp=0;

	/* compute traces */
	for (index_t i=0; i<m_2; ++i)
	{
		pp+=m_kernel->kernel(i, m_2+i);
		qq+=m_kernel->kernel(m+i, m+m_2+i);
		pq+=m_kernel->kernel(i, m+m_2+i);
		qp+=m_kernel->kernel(m_2+i, m+i);
	}

	SG_DEBUG("returning: 1/%d*(%f+%f-%f-%f)\n", m_2, pp, qq, pq, qp);

	/* mean of sum all traces is linear time mmd */
	SG_DEBUG("leaving CLinearTimeMMD::compute_statistic()\n");
	return 1.0/m_2*(pp+qq-pq-qp);
}

float64_t CLinearTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		if (m_p_and_q->get_num_vectors()<10000)
		{
			SG_WARNING("CLinearTimeMMD::compute_p_value: The number of samples"
					" should be very large (at least 10000)  in order to get a"
					" good Gaussian approximation using MMD1_GAUSSIAN.\n");
		}

		{
			/* compute variance and use to estimate Gaussian distribution */
			float64_t std_dev=CMath::sqrt(compute_variance_estimate());
			result=1.0-CStatistics::normal_cdf(statistic, std_dev);
		}
		break;
	default:
		result=CKernelTwoSampleTestStatistic::compute_p_value(statistic);
		break;
	}

	return result;
}

float64_t CLinearTimeMMD::compute_threshold(float64_t alpha)
{
	SG_ERROR("%s::compute_threshold is not yet implemented!\n");
	return 0;
}

float64_t CLinearTimeMMD::compute_variance_estimate()
{
	/* this corresponds to computing the statistic itself, however, the
	 * difference is that all terms (of the traces) have to be stored */
	index_t m=m_q_start;
	index_t m_2=m/2;

	m_kernel->init(m_p_and_q, m_p_and_q);

	/* allocate memory for traces */
	SGVector<float64_t> traces(m_2);

	/* sum up diagonals of all kernel matrices */
	for (index_t i=0; i<m_2; ++i)
	{
		/* init for code beauty :) */
		traces[i]=0;

		/* p and p */
		traces[i]+=m_kernel->kernel(i, m_2+i);

		/* q and q */
		traces[i]+=m_kernel->kernel(m+i, m+m_2+i);

		/* p and q */
		traces[i]-=m_kernel->kernel(i, m+m_2+i);

		/* q and p */
		traces[i]-=m_kernel->kernel(m_2+i, m+i);
	}

	/* return linear time variance estimate */
	return CStatistics::variance(traces)/m_2;
}

