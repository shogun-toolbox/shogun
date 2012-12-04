/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/MaxMMDKernelSelection.h>
#include <shogun/statistics/KernelTwoSampleTestStatistic.h>

using namespace shogun;

CMaxMMDKernelSelection::CMaxMMDKernelSelection() : CMMDKernelSelection()
{
}

CMaxMMDKernelSelection::CMaxMMDKernelSelection(
		CKernelTwoSampleTestStatistic* mmd) : CMMDKernelSelection(mmd)
{
}

CMaxMMDKernelSelection::~CMaxMMDKernelSelection()
{
}

float64_t CMaxMMDKernelSelection::compute_measure(CKernel* kernel)
{
	/* just return plain MMD */
	m_mmd->set_kernel(kernel);
	return m_mmd->compute_statistic();
}
