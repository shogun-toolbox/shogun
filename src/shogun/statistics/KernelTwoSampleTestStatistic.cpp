/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/KernelTwoSampleTestStatistic.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>

using namespace shogun;

CKernelTwoSampleTestStatistic::CKernelTwoSampleTestStatistic() :
		CTwoDistributionsTestStatistic()
{
	init();
}

CKernelTwoSampleTestStatistic::CKernelTwoSampleTestStatistic(CKernel* kernel,
		CFeatures* p_and_q, index_t q_start) :
		CTwoDistributionsTestStatistic(p_and_q, q_start)
{
	init();

	m_kernel=kernel;
	SG_REF(kernel);
}

CKernelTwoSampleTestStatistic::CKernelTwoSampleTestStatistic(CKernel* kernel,
		CFeatures* p, CFeatures* q) : CTwoDistributionsTestStatistic(p, q)
{
	init();

	m_kernel=kernel;
	SG_REF(kernel);
}

CKernelTwoSampleTestStatistic::~CKernelTwoSampleTestStatistic()
{
	SG_UNREF(m_kernel);
}

void CKernelTwoSampleTestStatistic::init()
{
	SG_ADD((CSGObject**)&m_kernel, "kernel", "Kernel for two sample test",
			MS_AVAILABLE);
	m_kernel=NULL;
}
