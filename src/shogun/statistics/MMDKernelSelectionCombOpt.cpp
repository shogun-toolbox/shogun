/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionCombOpt.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/CombinedKernel.h>


using namespace shogun;

CMMDKernelSelectionCombOpt::CMMDKernelSelectionCombOpt() :
		CMMDKernelSelectionComb()
{
}

CMMDKernelSelectionCombOpt::CMMDKernelSelectionCombOpt(
		CKernelTwoSampleTestStatistic* mmd, float64_t lambda) :
		CMMDKernelSelectionComb(mmd, lambda)
{
	/* currently, this method is only developed for the linear time MMD */
	REQUIRE(dynamic_cast<CLinearTimeMMD*>(mmd), "%s::%s(): Only "
			"CLinearTimeMMD is currently supported! Provided instance is "
			"\"%s\"\n", get_name(), get_name(), mmd->get_name());
}

CMMDKernelSelectionCombOpt::~CMMDKernelSelectionCombOpt()
{
}

#ifdef HAVE_LAPACK
SGVector<float64_t> CMMDKernelSelectionCombOpt::compute_measures()
{
	/* cast is safe due to assertion in constructor */
	CCombinedKernel* kernel=(CCombinedKernel*)m_mmd->get_kernel();
	index_t num_kernels=kernel->get_num_subkernels();
	SG_UNREF(kernel);

	/* allocate space for MMDs and Q matrix */
	SGVector<float64_t> mmds(num_kernels);
	m_Q=SGMatrix<float64_t>(num_kernels, num_kernels);

	/* online compute mmds and covariance matrix Q of kernels */
	((CLinearTimeMMD*)m_mmd)->compute_statistic_and_Q(mmds, m_Q);

	/* evtl regularize to avoid numerical problems (see NIPS paper) */
	if (m_lambda)
	{
		SG_DEBUG("regularizing matrix Q by adding %f to diagonal\n", m_lambda);
		for (index_t i=0; i<num_kernels; ++i)
			m_Q(i,i)+=m_lambda;
	}

	if (sg_io->get_loglevel()==MSG_DEBUG)
	{
		m_Q.display_matrix("(regularized) Q");
		mmds.display_vector("mmds");
	}

	/* solve the generated problem */
	SGVector<float64_t> result=solve_optimization(mmds);

	/* free matrix by hand since it is static */
	m_Q.~SGMatrix();

	return result;
}

#endif
