/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/statistics/KernelTwoSampleTestStatistic.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/statistics/QuadraticTimeMMD.h>


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
	REQUIRE(dynamic_cast<CLinearTimeMMD*>(mmd) ||
			dynamic_cast<CQuadraticTimeMMD*>(mmd),
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

SGVector<float64_t> CMMDKernelSelection::compute_measures()
{
	/* cast is safe due to assertion in constructor */
	CCombinedKernel* kernel=(CCombinedKernel*)m_mmd->get_kernel();
	/* compute measure for all kernels */
	SGVector<float64_t> measures(kernel->get_num_subkernels());
	CKernel* current=kernel->get_first_kernel();
	index_t i=0;
	while (current)
	{
		measures[i++]=compute_measure(current);
		SG_UNREF(current);
		current=kernel->get_next_kernel();
	}

	/* clean up */
	SG_UNREF(kernel);

	return measures;
}

CKernel* CMMDKernelSelection::select_kernel()
{
	SG_DEBUG("entering %s::select_kernel()\n", get_name());

	/* cast is safe due to assertion in constructor */
	CCombinedKernel* kernel=(CCombinedKernel*)m_mmd->get_kernel();

	/* compute measure for all kernels and remember yet largest */
	SGVector<float64_t> measures(kernel->get_num_subkernels());
	CKernel* current=kernel->get_first_kernel();
	CKernel* max_kernel=current;
	float64_t max_measure=-CMath::INFTY;
	index_t i=0;
	while (current)
	{
		/* compute measure and update maximum measure kernel */
		measures[i]=compute_measure(current);
		SG_PRINT("Computed measure for %s at %p: %f\n", current->get_name(),
				current, measures[i]);
		if (measures[i]>max_measure)
		{
			SG_PRINT("found new maximum!\n");
			max_measure=measures[i];
			max_kernel=current;
		}

		/* proceed to next */
		SG_UNREF(current);
		current=kernel->get_next_kernel();
		++i;
	}

	SG_DEBUG("leaving %s::select_kernel()\n", get_name());

	/* increase refcount of resulting kernel and return */
	SG_REF(max_kernel);

	/* clean up */
	SG_UNREF(kernel);

	return max_kernel;
}

