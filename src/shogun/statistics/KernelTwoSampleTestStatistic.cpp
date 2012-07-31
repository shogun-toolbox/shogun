/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/KernelTwoSampleTestStatistic.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>

using namespace shogun;

CKernelTwoSampleTestStatistic::CKernelTwoSampleTestStatistic() :
		CTwoDistributionsTestStatistic()
{
	init();
}

CKernelTwoSampleTestStatistic::CKernelTwoSampleTestStatistic(CKernel* kernel,
		CFeatures* p_and_q, index_t q_start) :
		CTwoDistributionsTestStatistic(p_and_q, q_start)
{
	init();

	m_kernel=kernel;
	SG_REF(kernel);
}

CKernelTwoSampleTestStatistic::CKernelTwoSampleTestStatistic(CKernel* kernel,
		CFeatures* p, CFeatures* q) : CTwoDistributionsTestStatistic(p, q)
{
	init();

	m_kernel=kernel;
	SG_REF(kernel);
}

CKernelTwoSampleTestStatistic::~CKernelTwoSampleTestStatistic()
{
	SG_UNREF(m_kernel);
}

void CKernelTwoSampleTestStatistic::init()
{
	SG_ADD((CSGObject**)&m_kernel, "kernel", "Kernel for two sample test",
			MS_AVAILABLE);
	m_kernel=NULL;
}

SGVector<float64_t> CKernelTwoSampleTestStatistic::bootstrap_null()
{
	/* compute bootstrap statistics for null distribution */
	SGVector<float64_t> results(m_bootstrap_iterations);

	/* memory for index permutations, (would slow down loop) */
	SGVector<index_t> ind_permutation(m_p_and_q->get_num_vectors());
	ind_permutation.range_fill();

	/* check if kernel is a custom kernel. In that case, changing features is
	 * not what we want but just subsetting the kernel itself */
	CCustomKernel* custom_kernel;
	if (m_kernel->get_kernel_type()==K_CUSTOM)
	{
		custom_kernel=(CCustomKernel*)m_kernel;
		custom_kernel->add_row_subset(ind_permutation);
		custom_kernel->add_col_subset(ind_permutation);
	}
	else
	{
		custom_kernel=NULL;
		m_p_and_q->add_subset(ind_permutation);
	}


	for (index_t i=0; i<m_bootstrap_iterations; ++i)
	{
		/* idea: merge features of p and q, shuffle, and compute statistic.
		 * This is done using subsets here */

		/* create index permutation and add as subset. This will mix samples
		 * from p and q */
		SGVector<int32_t>::permute_vector(ind_permutation);

		/* compute statistic for this permutation of mixed samples */
		results[i]=compute_statistic();
	}

	/* clean up */
	if (custom_kernel)
	{
		custom_kernel->remove_row_subset();
		custom_kernel->remove_col_subset();
	}
	else
	{
		custom_kernel=NULL;
		m_p_and_q->remove_subset();
	}

	/* clean up and return */
	return results;
}
