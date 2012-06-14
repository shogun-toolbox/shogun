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

CLinearTimeMMD::CLinearTimeMMD() : CKernelTwoSampleTestStatistic()
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
	/* TODO register parameters*/
}

float64_t CLinearTimeMMD::compute_statistic()
{
	/* TODO maybe add parallelized kernel matrix trace method to CKernel? */
	/* TODO features with a different number of vectors should be allowed */

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	// TODO here is something wrong! (possibly)
	index_t m=m_q_start;
	index_t m_2=m/2;

	/* allocate memory */
	SGVector<float64_t> tr_K_x(m_2);
	SGVector<float64_t> tr_K_y(m_2);
	SGVector<float64_t> tr_K_xy(m);

	/* compute traces of kernel matrices for linear MMD */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* p and p */
	for (index_t i=0; i<m_2; ++i)
		tr_K_x.vector[i]=m_kernel->kernel(i, m_2+i);

	/* q and q */
	for (index_t i=m_q_start; i<m+m_2; ++i)
		tr_K_y.vector[i-m_q_start]=m_kernel->kernel(i, m_2+i);

	/* p and q */
	for (index_t i=0; i<m; ++i)
		tr_K_xy.vector[i]=m_kernel->kernel(i, m+i);

	/* compute result */
	float64_t first=0;
	float64_t second=0;
	float64_t third=0;

	for (index_t i=0; i<m_2; ++i)
	{
		first+=tr_K_x.vector[i];
		second+=tr_K_y.vector[i];
		third+=tr_K_xy.vector[i];
	}

	for (index_t i=m_2; i<m; ++i)
		third+=tr_K_xy.vector[i-m_2];

	return 1.0/m_2*(first+second)+1.0/m*third;
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

