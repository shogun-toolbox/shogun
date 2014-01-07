/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <statistics/MMDKernelSelectionOpt.h>
#include <statistics/LinearTimeMMD.h>
#include <kernel/CombinedKernel.h>

using namespace shogun;

CMMDKernelSelectionOpt::CMMDKernelSelectionOpt() :
		CMMDKernelSelection()
{
	init();
}

CMMDKernelSelectionOpt::CMMDKernelSelectionOpt(
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

CMMDKernelSelectionOpt::~CMMDKernelSelectionOpt()
{
}

SGVector<float64_t> CMMDKernelSelectionOpt::compute_measures()
{
	/* comnpute mmd on all subkernels using the same data. Note that underlying
	 * kernel was asserted to be a combined one */
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	((CLinearTimeMMD*)m_mmd)->compute_statistic_and_variance(mmds, vars, true);

	/* we know that the underlying MMD is linear time version, cast is safe */
	SGVector<float64_t> measures(mmds.vlen);

	for (index_t i=0; i<measures.vlen; ++i)
		measures[i]=mmds[i]/(CMath::sqrt(vars[i])+m_lambda);

	return measures;
}

void CMMDKernelSelectionOpt::init()
{
	/* set to a sensible standard value that proved to be useful in
	 * experiments, see NIPS paper */
	m_lambda=1E-5;
}
