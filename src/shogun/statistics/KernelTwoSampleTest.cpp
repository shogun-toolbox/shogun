/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/KernelTwoSampleTest.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CKernelTwoSampleTest::CKernelTwoSampleTest() :
		CTwoSampleTest()
{
	init();
}

CKernelTwoSampleTest::CKernelTwoSampleTest(CKernel* kernel,
		CFeatures* p_and_q, index_t q_start) :
		CTwoSampleTest(p_and_q, q_start)
{
	init();

	m_kernel=kernel;
	SG_REF(kernel);
}

CKernelTwoSampleTest::CKernelTwoSampleTest(CKernel* kernel,
		CFeatures* p, CFeatures* q) : CTwoSampleTest(p, q)
{
	init();

	m_kernel=kernel;
	SG_REF(kernel);
}

CKernelTwoSampleTest::~CKernelTwoSampleTest()
{
	SG_UNREF(m_kernel);
}

void CKernelTwoSampleTest::init()
{
	SG_ADD((CSGObject**)&m_kernel, "kernel", "Kernel for two sample test",
			MS_AVAILABLE);
	m_kernel=NULL;
}

SGVector<float64_t> CKernelTwoSampleTest::sample_null()
{
	SG_DEBUG("entering!\n");

	REQUIRE(m_kernel, "No kernel set!\n");
	REQUIRE(m_kernel->get_kernel_type()==K_CUSTOM || m_p_and_q,
			"No features and no custom kernel set!\n");

	/* compute sample statistics for null distribution */
	SGVector<float64_t> results;

	/* only do something if a custom kernel is used: use the power of pre-
	 * computed kernel matrices
	 */
	if (m_kernel->get_kernel_type()==K_CUSTOM)
	{
		/* allocate memory */
		results=SGVector<float64_t>(m_num_null_samples);

		/* in case of custom kernel, there are no features */
		index_t num_data;
		if (m_kernel->get_kernel_type()==K_CUSTOM)
			num_data=m_kernel->get_num_vec_lhs();
		else
			num_data=m_p_and_q->get_num_vectors();

		/* memory for index permutations, (would slow down loop) */
		SGVector<index_t> ind_permutation(num_data);
		ind_permutation.range_fill();

		/* check if kernel is a custom kernel. In that case, changing features is
		 * not what we want but just subsetting the kernel itself */
		CCustomKernel* custom_kernel=(CCustomKernel*)m_kernel;

		for (index_t i=0; i<m_num_null_samples; ++i)
		{
			/* idea: merge features of p and q, shuffle, and compute statistic.
			 * This is done using subsets here. add to custom kernel since
			 * it has no features to subset. CustomKernel has not to be
			 * re-initialised after each subset setting */
			CMath::permute(ind_permutation);

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
		results=CTwoSampleTest::sample_null();
	}

	SG_DEBUG("leaving!\n");

	return results;
}
