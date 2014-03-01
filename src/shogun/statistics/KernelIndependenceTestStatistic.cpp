/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/KernelIndependenceTestStatistic.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>

using namespace shogun;

CKernelIndependenceTestStatistic::CKernelIndependenceTestStatistic() :
		CIndependenceTestStatistic()
{
	init();
}

CKernelIndependenceTestStatistic::CKernelIndependenceTestStatistic(
		CKernel* kernel_p, CKernel* kernel_q, CFeatures* p_and_q,
		index_t q_start) : CIndependenceTestStatistic(m_p_and_q, q_start)
{
	init();

	m_kernel_p=kernel_p;
	m_kernel_q=kernel_q;
	SG_REF(kernel_p);
	SG_REF(kernel_q);
}

CKernelIndependenceTestStatistic::CKernelIndependenceTestStatistic(
		CKernel* kernel_p, CKernel* kernel_q, CFeatures* p, CFeatures* q) :
		CIndependenceTestStatistic(p, q)
{
	init();

	m_kernel_p=kernel_p;
	SG_REF(kernel_p);

	m_kernel_q=kernel_q;
	SG_REF(kernel_q);
}

CKernelIndependenceTestStatistic::~CKernelIndependenceTestStatistic()
{
	SG_UNREF(m_kernel_p);
	SG_UNREF(m_kernel_q);
}

void CKernelIndependenceTestStatistic::init()
{
	SG_ADD((CSGObject**)&m_kernel_p, "kernel_p", "Kernel for samples from p",
			MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_kernel_q, "kernel_q", "Kernel for samples from q",
			MS_AVAILABLE);
	m_kernel_p=NULL;
	m_kernel_q=NULL;
}

SGVector<float64_t> CKernelIndependenceTestStatistic::sample_null()
{
	SG_DEBUG("entering CKernelIndependenceTestStatistic::sample_null()\n")

	/* compute sample statistics for null distribution */
	SGVector<float64_t> results;

	/* only do something if a custom kernel is used: use the power of pre-
	 * computed kernel matrices
	 */
	if (m_kernel_p->get_kernel_type()==K_CUSTOM &&
			m_kernel_q->get_kernel_type()==K_CUSTOM)
	{
		/* allocate memory */
		results=SGVector<float64_t>(m_num_permutation_iterations);

		/* memory for index permutations for only kernel matrix for p */
		SGVector<index_t> ind_permutation(m_m);
		ind_permutation.range_fill();

		/* check if kernel is a custom kernel. In that case, changing features is
		 * not what we want but just subsetting the kernel itself */
		CCustomKernel* custom_kernel_p=(CCustomKernel*)m_kernel_p;

		for (index_t i=0; i<m_num_permutation_iterations; ++i)
		{
			/* idea: merge features of p and q, shuffle samples from p while
			 * keeping samples from q intact and compute statistic.
			 * This is done using subsets here. add to custom kernel since
			 * it has no features to subset. CustomKernel has not to be
			 * re-initialised after each subset setting */
			SGVector<int32_t>::permute_vector(ind_permutation);

			custom_kernel_p->add_row_subset(ind_permutation);
			custom_kernel_p->add_col_subset(ind_permutation);

			/* compute statistic for this permutation of mixed samples */
			results[i]=compute_statistic();

			/* remove subsets */
			custom_kernel_p->remove_row_subset();
			custom_kernel_p->remove_col_subset();
		}
	}
	else
	{
		/* in this case, just use superclass method */
		results=CIndependenceTestStatistic::sample_null();
	}


	SG_DEBUG("leaving CKernelIndependenceTestStatistic::sample_null()\n")
	return results;
}

