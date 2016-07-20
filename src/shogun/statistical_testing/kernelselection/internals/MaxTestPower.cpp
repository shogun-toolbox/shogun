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
#include <shogun/statistical_testing/StreamingMMD.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/MultiKernelQuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxTestPower.h>

using namespace shogun;
using namespace internal;

MaxTestPower::MaxTestPower(KernelManager& km, CMMD* est) : MaxMeasure(km, est), lambda(1E-5)
{
}

MaxTestPower::~MaxTestPower()
{
}

void MaxTestPower::compute_measures()
{
	init_measures();
	REQUIRE(estimator!=nullptr, "Estimator is not set!\n");
	const auto m=estimator->get_num_samples_p();
	const auto n=estimator->get_num_samples_q();
	auto existing_kernel=estimator->get_kernel();
	const size_t num_kernels=kernel_mgr.num_kernels();
	auto streaming_mmd=dynamic_cast<CStreamingMMD*>(estimator);
	if (streaming_mmd)
	{
		for (size_t i=0; i<num_kernels; ++i)
		{
			auto kernel=kernel_mgr.kernel_at(i);
			estimator->set_kernel(kernel);
			auto estimates=streaming_mmd->compute_statistic_variance();
			auto var_est=estimates.first;
			auto mmd_est=estimates.second*(m+n)/m/n;
			measures[i]=var_est/CMath::sqrt(mmd_est+lambda);
			estimator->cleanup();
		}
	}
	else
	{
		auto quadratictime_mmd=dynamic_cast<CQuadraticTimeMMD*>(estimator);
		ASSERT(quadratictime_mmd);
		measures=quadratictime_mmd->multikernel()->test_power(kernel_mgr);
	}
	if (existing_kernel)
		estimator->set_kernel(existing_kernel);
}
