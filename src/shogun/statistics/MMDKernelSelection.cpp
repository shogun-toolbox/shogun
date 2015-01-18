/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/statistics/MMDKernelSelection.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/statistics/KernelTwoSampleTest.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/statistics/QuadraticTimeMMD.h>

using namespace shogun;

CMMDKernelSelection::CMMDKernelSelection()
{
}

CMMDKernelSelection::CMMDKernelSelection(CKernelTwoSampleTest* mmd)
	: CKernelSelection(mmd)
{
	/* ensure that mmd contains an instance of a MMD related class
	   TODO - Add S_BTEST_MMD when feature/mmd is merged with develop */
	REQUIRE(mmd->get_statistic_type()==S_LINEAR_TIME_MMD ||
			mmd->get_statistic_type()==S_QUADRATIC_TIME_MMD,
			"Provided instance for kernel two sample testing has to be a MMD-"
			"based class! The provided is of class \"%s\"\n", mmd->get_name());
}

CMMDKernelSelection::~CMMDKernelSelection()
{
}

CKernel* CMMDKernelSelection::select_kernel()
{
	SG_DEBUG("entering\n")

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
	CCombinedKernel* combined=(CCombinedKernel*)m_estimator->get_kernel();
	CKernel* current=combined->get_kernel(max_idx);

	SG_UNREF(combined);
	SG_DEBUG("leaving\n");

	/* current is not SG_UNREF'ed nor SG_REF'ed since the counter needs to be
	 * incremented exactly by one */
	return current;
}

