/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#include <algorithm>
#include <shogun/lib/SGVector.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/MaxTestPower.h>
#include <shogun/statistical_testing/internals/KernelManager.h>

using namespace shogun;
using namespace internal;

MaxTestPower::MaxTestPower(KernelManager& km, CMMD* est) : KernelSelection(km), estimator(est), lambda(1E-5)
{
}

MaxTestPower::~MaxTestPower()
{
}

SGVector<float64_t> MaxTestPower::compute_measures()
{
	REQUIRE(estimator!=nullptr, "Estimator is not set!\n");
	REQUIRE(kernel_mgr.num_kernels()>0, "Number of kernels is %d!\n", kernel_mgr.num_kernels());

	SGVector<float64_t> result(kernel_mgr.num_kernels());
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
	{
		auto kernel=kernel_mgr.kernel_at(i);
		estimator->set_kernel(kernel);
		auto estimates=estimator->compute_statistic_variance();
		result[i]=estimates.first/CMath::sqrt(estimates.second+lambda);
		estimator->cleanup();
	}
	return result;
}

CKernel* MaxTestPower::select_kernel()
{
	SGVector<float64_t> measures=compute_measures();
	auto max_element=std::max_element(measures.vector, measures.vector+measures.vlen);
	auto max_idx=std::distance(measures.vector, max_element);
	SG_SDEBUG("Selected kernel at %d position!\n", max_idx);
	return kernel_mgr.kernel_at(max_idx);
}
