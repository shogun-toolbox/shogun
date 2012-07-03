/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/KernelIndependenceTestStatistic.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>

using namespace shogun;

CKernelIndependenceTestStatistic::CKernelIndependenceTestStatistic() :
		CIndependenceTestStatistic()
{
	init();
}

CKernelIndependenceTestStatistic::CKernelIndependenceTestStatistic(
		CKernel* kernel_p, CKernel* kernel_q, CFeatures* p, CFeatures* q) :
		CIndependenceTestStatistic(p, q)
{
	init();

	m_kernel_p=kernel_p;
	m_kernel_q=kernel_q;
	SG_REF(kernel_p);
	SG_REF(kernel_q);
}

CKernelIndependenceTestStatistic::~CKernelIndependenceTestStatistic()
{
	SG_UNREF(m_kernel_p);
	SG_UNREF(m_kernel_q);
}

void CKernelIndependenceTestStatistic::init()
{
	SG_ADD((CSGObject**)&m_kernel_p, "kernel_p", "Kernel for samples from p",
			MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_kernel_q, "kernel_q", "Kernel for samples from q",
			MS_AVAILABLE);
	m_kernel_p=NULL;
	m_kernel_q=NULL;
}
