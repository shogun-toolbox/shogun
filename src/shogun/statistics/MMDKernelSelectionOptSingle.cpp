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

float64_t CMMDKernelSelectionOptSingle::compute_measure(CKernel* kernel)
{
	/* compute MMD and its standard deviation estimate on given kernel */
	m_mmd->set_kernel(kernel);

	/* we know that the underlying MMD is linear time version, cast is safe */
	float64_t statistic;
	float64_t variance;
	((CLinearTimeMMD*)m_mmd)->compute_statistic_and_variance(statistic,
			variance);

	return statistic/(CMath::sqrt(variance)+m_lambda);
}

void CMMDKernelSelectionOptSingle::init()
{
	/* set to a sensible standard value that proved to be useful in
	 * experiments, see NIPS paper */
	m_lambda=10E-5;
}
