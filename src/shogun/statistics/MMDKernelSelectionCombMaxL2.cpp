/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionCombMaxL2.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/mathematics/Statistics.h>


using namespace shogun;

CMMDKernelSelectionCombMaxL2::CMMDKernelSelectionCombMaxL2() :
		CMMDKernelSelectionComb()
{
}

CMMDKernelSelectionCombMaxL2::CMMDKernelSelectionCombMaxL2(
		CKernelTwoSampleTest* mmd) : CMMDKernelSelectionComb(mmd)
{
	/* currently, this method is only developed for the linear time MMD */
	REQUIRE(mmd->get_statistic_type()==S_QUADRATIC_TIME_MMD ||
			mmd->get_statistic_type()==S_LINEAR_TIME_MMD, "%s::%s(): Only "
			"CLinearTimeMMD is currently supported! Provided instance is "
			"\"%s\"\n", get_name(), get_name(), mmd->get_name());
}

CMMDKernelSelectionCombMaxL2::~CMMDKernelSelectionCombMaxL2()
{
}

#ifdef HAVE_LAPACK
SGVector<float64_t> CMMDKernelSelectionCombMaxL2::compute_measures()
{
	/* cast is safe due to assertion in constructor */
	CCombinedKernel* kernel=(CCombinedKernel*)m_mmd->get_kernel();
	index_t num_kernels=kernel->get_num_subkernels();
	SG_UNREF(kernel);

	/* compute mmds for all underlying kernels and create identity matrix Q
	 * (see NIPS paper) */
	SGVector<float64_t> mmds=m_mmd->compute_statistic(true);

	/* free matrix by hand since it is static */
	SG_FREE(m_Q.matrix);
	m_Q.matrix=NULL;
	m_Q.num_rows=0;
	m_Q.num_cols=0;
	m_Q=SGMatrix<float64_t>(num_kernels, num_kernels, false);
	for (index_t i=0; i<num_kernels; ++i)
	{
		for (index_t j=0; j<num_kernels; ++j)
			m_Q(i, j)=i==j ? 1 : 0;
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
#endif
