/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionMax.h>
#include <shogun/statistics/KernelTwoSampleTest.h>

using namespace shogun;

CMMDKernelSelectionMax::CMMDKernelSelectionMax() : CMMDKernelSelection()
{
}

CMMDKernelSelectionMax::CMMDKernelSelectionMax(
		CKernelTwoSampleTest* mmd) : CMMDKernelSelection(mmd)
{
}

CMMDKernelSelectionMax::~CMMDKernelSelectionMax()
{
}

SGVector<float64_t> CMMDKernelSelectionMax::compute_measures()
{
	/* simply return vector with MMDs */
	return m_mmd->compute_statistic(true);
}
