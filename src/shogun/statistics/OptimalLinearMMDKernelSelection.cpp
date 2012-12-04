/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/OptimalLinearMMDKernelSelection.h>
#include <shogun/statistics/LinearTimeMMD.h>


using namespace shogun;

COptimalLinearMMDKernelSelection::COptimalLinearMMDKernelSelection() :
		CMMDKernelSelection()
{
	init();
}

COptimalLinearMMDKernelSelection::COptimalLinearMMDKernelSelection(
		CLinearTimeMMD* mmd, float64_t lambda) :
		CMMDKernelSelection((CKernelTwoSampleTestStatistic*)mmd)
{
	init();

	m_lambda=lambda;
}

COptimalLinearMMDKernelSelection::~COptimalLinearMMDKernelSelection()
{
}

float64_t COptimalLinearMMDKernelSelection::compute_measure(CKernel* kernel)
{
	/* compute MMD and its standard deviation estimate on given kernel */
	m_mmd->set_kernel(kernel);

	/* we know that the underlying MMD is linear time version, cast is safe */
	float64_t statistic;
	float64_t variance;
	((CLinearTimeMMD*)m_mmd)->compute_statistic_and_variance(statistic, variance);

	return statistic/(CMath::sqrt(variance)+m_lambda);
}

void COptimalLinearMMDKernelSelection::init()
{
	/* set to a sensible standard value that proved to be useful in
	 * experiments
	 */
	m_lambda=10E-5;
}
