/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <statistics/MMDKernelSelection.h>
#include <kernel/CombinedKernel.h>
#include <statistics/KernelTwoSampleTestStatistic.h>
#include <statistics/LinearTimeMMD.h>
#include <statistics/QuadraticTimeMMD.h>


using namespace shogun;

CMMDKernelSelection::CMMDKernelSelection()
{
	init();
}

CMMDKernelSelection::CMMDKernelSelection(
		CKernelTwoSampleTestStatistic* mmd)
{
	init();

	/* ensure that mmd contains an instance of a MMD related class */
	REQUIRE(mmd, "CMMDKernelSelection::CMMDKernelSelection(): No MMD instance "
			"provided!\n");
	REQUIRE(mmd->get_statistic_type()==S_LINEAR_TIME_MMD ||
			mmd->get_statistic_type()==S_QUADRATIC_TIME_MMD,
			"CMMDKernelSelection::CMMDKernelSelection(): provided instance "
			"for kernel two sample testing has to be a MMD-based class! The "
			"provided is of class \"%s\"\n", mmd->get_name());

	/* ensure that there is a combined kernel */
	CKernel* kernel=mmd->get_kernel();
	REQUIRE(kernel, "CMMDKernelSelection::CMMDKernelSelection(): underlying "
			"\"%s\" has no kernel set!\n", mmd->get_name());
	REQUIRE(kernel->get_kernel_type()==K_COMBINED, "CMMDKernelSelection::"
			"CMMDKernelSelection(): kernel of underlying \"%s\" is of type \"%s\""
			" but is has to be CCombinedKernel\n", mmd->get_name(),
			kernel->get_name());
	SG_UNREF(kernel);

	m_mmd=mmd;
	SG_REF(m_mmd);
}


CMMDKernelSelection::~CMMDKernelSelection()
{
	SG_UNREF(m_mmd);
}

void CMMDKernelSelection::init()
{
	m_mmd=NULL;

	SG_ADD((CSGObject**)&m_mmd, "mmd", "Underlying MMD instance",
			MS_NOT_AVAILABLE);
}

CKernel* CMMDKernelSelection::select_kernel()
{
	SG_DEBUG("entering CMMDKernelSelection::select_kernel()\n")

	/* compute measures and return single kernel with maximum measure */
	SGVector<float64_t> measures=compute_measures();

	/* find maximum and return corresponding kernel */
	float64_t max=measures[0];
	index_t max_idx=0;
	for (index_t i=1; i<measures.vlen; ++i)
	{
		if (measures[i]>max)
		{
			max=measures[i];
			max_idx=i;
		}
	}

	/* find kernel with corresponding index */
	CCombinedKernel* combined=(CCombinedKernel*)m_mmd->get_kernel();
	CKernel* current=combined->get_kernel(max_idx);

	SG_UNREF(combined);
	SG_DEBUG("leaving CMMDKernelSelection::select_kernel()\n");

	/* current is not SG_UNREF'ed nor SG_REF'ed since the counter needs to be
	 * incremented exactly by one */
	return current;
}

