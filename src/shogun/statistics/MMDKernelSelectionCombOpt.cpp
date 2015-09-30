/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionCombOpt.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/CombinedKernel.h>


using namespace shogun;

CMMDKernelSelectionCombOpt::CMMDKernelSelectionCombOpt() :
		CMMDKernelSelectionComb()
{
	init();
}

CMMDKernelSelectionCombOpt::CMMDKernelSelectionCombOpt(
		CKernelTwoSampleTest* mmd, float64_t lambda) :
		CMMDKernelSelectionComb(mmd)
{
	/* currently, this method is only developed for the linear time MMD */
	REQUIRE(dynamic_cast<CLinearTimeMMD*>(mmd), "%s::%s(): Only "
			"CLinearTimeMMD is currently supported! Provided instance is "
			"\"%s\"\n", get_name(), get_name(), mmd->get_name());

	init();

	m_lambda=lambda;
}

CMMDKernelSelectionCombOpt::~CMMDKernelSelectionCombOpt()
{
}

void CMMDKernelSelectionCombOpt::init()
{
	/* set to a sensible standard value that proved to be useful in
	 * experiments, see NIPS paper */
	m_lambda=1E-5;

	SG_ADD(&m_lambda, "lambda", "Regularization parameter lambda",
			MS_NOT_AVAILABLE);
}

SGVector<float64_t> CMMDKernelSelectionCombOpt::compute_measures()
{
	/* cast is safe due to assertion in constructor */
	CCombinedKernel* kernel=(CCombinedKernel*)m_estimator->get_kernel();
	index_t num_kernels=kernel->get_num_subkernels();
	SG_UNREF(kernel);

	/* allocate space for MMDs and Q matrix */
	SGVector<float64_t> mmds(num_kernels);

	/* free matrix by hand since it is static */
	SG_FREE(m_Q.matrix);
	m_Q.matrix=NULL;
	m_Q.num_rows=0;
	m_Q.num_cols=0;
	m_Q=SGMatrix<float64_t>(num_kernels, num_kernels, false);

	/* online compute mmds and covariance matrix Q of kernels */
	((CLinearTimeMMD*)m_estimator)->compute_statistic_and_Q(mmds, m_Q);

	/* evtl regularize to avoid numerical problems (see NIPS paper) */
	if (m_lambda)
	{
		SG_DEBUG("regularizing matrix Q by adding %f to diagonal\n", m_lambda)
		for (index_t i=0; i<num_kernels; ++i)
			m_Q(i,i)+=m_lambda;
	}

	if (sg_io->get_loglevel()==MSG_DEBUG)
	{
		m_Q.display_matrix("(regularized) Q");
		mmds.display_vector("mmds");
	}

	/* solve the generated problem */
	SGVector<float64_t> result=CMMDKernelSelectionComb::solve_optimization(mmds);

	/* free matrix by hand since it is static (again) */
	SG_FREE(m_Q.matrix);
	m_Q.matrix=NULL;
	m_Q.num_rows=0;
	m_Q.num_cols=0;

	return result;
}

