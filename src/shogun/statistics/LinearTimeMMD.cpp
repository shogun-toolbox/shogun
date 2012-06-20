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

	if (q_start!=p_and_q->get_num_vectors()/2)
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
	/* TODO features with a different number of vectors should be allowed */

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m=m_q_start;
	index_t m_2=m/2;

	/* compute traces of kernel matrices for linear MMD */
	m_kernel->init(m_p_and_q, m_p_and_q);

	float64_t pp=0;
	float64_t qq=0;
	float64_t pq=0;
	float64_t qp=0;

	/* p and p, q and q, p and q first half */
	for (index_t i=0; i<m_2; ++i)
	{
		pp+=m_kernel->kernel(i, m_2+i);
		qq+=m_kernel->kernel(m+i, m+m_2+i);
		pq+=m_kernel->kernel(i, m+m_2+i);
		qp+=m_kernel->kernel(m_2+i, m+i);
	}

	/* mean of sum all traces is linear time mmd */
	return 1.0/m_2*(pp+qq-pq-qp);
}

float64_t CLinearTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_p_value_method)
	{
	/* TODO implement new null distribution approximations here */
	default:
		result=CKernelTwoSampleTestStatistic::compute_p_value(statistic);
		break;
	}

	return result;
}

