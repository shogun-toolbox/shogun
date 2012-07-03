/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/HSIC.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/kernel/Kernel.h>

using namespace shogun;

CHSIC::CHSIC() : CKernelIndependenceTestStatistic()
{
	init();
}

CHSIC::CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p,CFeatures* q) :
		CKernelIndependenceTestStatistic(kernel_p, kernel_q, p, q)
{
	if (p->get_num_vectors()!=q->get_num_vectors())
	{
		SG_ERROR("%s: Only features with equal number of vectors "
				"are currently possible\n", get_name());
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
	if (!m_kernel_p || m_kernel_q)
	{
		SG_ERROR("%s::compute_statistic(): No or only one kernel specified!\n",
				get_name());
	}

	/* compute kernel matrices (these have to be stored unfortunately */
	m_kernel_p->init(m_p, m_p);
	m_kernel_q->init(m_q, m_q);

	SGMatrix<float64_t> K=m_kernel_p->get_kernel_matrix();
	SGMatrix<float64_t> L=m_kernel_p->get_kernel_matrix();

	/* center matrices (replaces this H matrix from the paper) */
	K.center();
	L.center();

	/* compute MATLAB: sum(sum((H*K)' .* (H*L))), which is biased HSIC */
	index_t m=K.num_rows;
	float64_t result=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
			result+=K(j,i)*L(i,j);
	}

	result/=m*m;

	return result;
}

float64_t CHSIC::compute_p_value(float64_t statistic)
{
	return 0;
}

float64_t CHSIC::compute_threshold(float64_t alpha)
{
	return 0;
}
