/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionOptSingle.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/CombinedKernel.h>

using namespace shogun;

CMMDKernelSelectionOptSingle::CMMDKernelSelectionOptSingle() :
		CMMDKernelSelection()
{
	init();
}

CMMDKernelSelectionOptSingle::CMMDKernelSelectionOptSingle(
		CKernelTwoSampleTestStatistic* mmd, float64_t lambda) :
		CMMDKernelSelection(mmd)
{
	init();

	/* currently, this method is only developed for the linear time MMD */
	REQUIRE(dynamic_cast<CLinearTimeMMD*>(mmd), "%s::%s(): Only "
			"CLinearTimeMMD is currently supported! Provided instance is "
			"\"%s\"\n", get_name(), get_name(), mmd->get_name());

	m_lambda=lambda;
}

CMMDKernelSelectionOptSingle::~CMMDKernelSelectionOptSingle()
{
}

SGVector<float64_t> CMMDKernelSelectionOptSingle::compute_measures()
{
	/* create combined kernel to compute mmds on */
	CCombinedKernel* combined=new CCombinedKernel();
	CKernel* current=(CCombinedKernel*)m_kernel_list->get_first_element();
	while(current)
	{
		combined->append_kernel(current);
		SG_UNREF(current);
		current=(CCombinedKernel*)m_kernel_list->get_next_element();
	}

	/* comnpute mmd on all subkernels of combined kernel. This is done in order
	 * to compute the mmds all on the same data */
	m_mmd->set_kernel(combined);
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	((CLinearTimeMMD*)m_mmd)->compute_statistic_and_variance(mmds, vars, true);

	/* we know that the underlying MMD is linear time version, cast is safe */
	SGVector<float64_t> measures(mmds.vlen);

	for (index_t i=0; i<measures.vlen; ++i)
		measures[i]=mmds[i]/(vars[i]+m_lambda);

	return measures;
}

void CMMDKernelSelectionOptSingle::init()
{
	/* set to a sensible standard value that proved to be useful in
	 * experiments, see NIPS paper */
	m_lambda=10E-5;
}
