/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/HSIC.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/kernel/Kernel.h>

using namespace shogun;

CHSIC::CHSIC() : CKernelIndependenceTestStatistic()
{
	init();
}

CHSIC::CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p,CFeatures* q) :
		CKernelIndependenceTestStatistic(kernel_p, kernel_q, p, q)
{
	init();
}


CHSIC::~CHSIC()
{

}

void CHSIC::init()
{

}

float64_t CHSIC::compute_statistic()
{
	if (!m_kernel_p || m_kernel_q)
	{
		SG_ERROR("%s::compute_statistic(): No or only one kernel specified!\n",
				get_name());
	}

	return 0;
}

float64_t CHSIC::compute_p_value(float64_t statistic)
{
	return 0;
}

float64_t CHSIC::compute_threshold(float64_t alpha)
{
	return 0;
}
