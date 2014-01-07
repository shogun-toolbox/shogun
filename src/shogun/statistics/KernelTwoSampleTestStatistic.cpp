/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <statistics/KernelTwoSampleTestStatistic.h>
#include <features/Features.h>
#include <kernel/Kernel.h>
#include <kernel/CustomKernel.h>

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
	REQUIRE(m_kernel, "%s::bootstrap_null(): No kernel set!\n", get_name());
	REQUIRE(m_kernel->get_kernel_type()==K_CUSTOM || m_p_and_q,
			"%s::bootstrap_null(): No features and no custom kernel set!\n",
			get_name());

	/* compute bootstrap statistics for null distribution */
	SGVector<float64_t> results;

	/* only do something if a custom kernel is used: use the power of pre-
	 * computed kernel matrices
	 */
	if (m_kernel->get_kernel_type()==K_CUSTOM)
	{
		/* allocate memory */
		results=SGVector<float64_t>(m_bootstrap_iterations);

		/* memory for index permutations, (would slow down loop) */

		/* in case of custom kernel, there are no features */
		index_t num_data;
		if (m_kernel->get_kernel_type()==K_CUSTOM)
			num_data=m_kernel->get_num_vec_lhs();
		else
			num_data=m_p_and_q->get_num_vectors();

		SGVector<index_t> ind_permutation(num_data);
		ind_permutation.range_fill();

		/* check if kernel is a custom kernel. In that case, changing features is
		 * not what we want but just subsetting the kernel itself */
		CCustomKernel* custom_kernel=(CCustomKernel*)m_kernel;

		for (index_t i=0; i<m_bootstrap_iterations; ++i)
		{
			/* idea: merge features of p and q, shuffle, and compute statistic.
			 * This is done using subsets here. add to custom kernel since
			 * it has no features to subset. CustomKernel has not to be
			 * re-initialised after each subset setting */
			SGVector<int32_t>::permute_vector(ind_permutation);

			custom_kernel->add_row_subset(ind_permutation);
			custom_kernel->add_col_subset(ind_permutation);

			/* compute statistic for this permutation of mixed samples */
			results[i]=compute_statistic();

			/* remove subsets */
			custom_kernel->remove_row_subset();
			custom_kernel->remove_col_subset();
		}
	}
	else
	{
		/* in this case, just use superclass method */
		results=CTwoDistributionsTestStatistic::bootstrap_null();
	}

	return results;
}
