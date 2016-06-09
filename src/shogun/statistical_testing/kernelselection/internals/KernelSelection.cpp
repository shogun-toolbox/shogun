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

#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/ShiftInvariantKernel.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/kernelselection/internals/KernelSelection.h>

using namespace shogun;
using namespace internal;

KernelSelection::KernelSelection(KernelManager& km, CMMD* est) : kernel_mgr(km), estimator(est)
{
	REQUIRE(kernel_mgr.num_kernels()>0, "Number of kernels is %d!\n", kernel_mgr.num_kernels());
	REQUIRE(estimator!=nullptr, "Estimator is not set!\n");
}

KernelSelection::~KernelSelection()
{
}

bool KernelSelection::same_distance_type() const
{
	bool same=false;
	EDistanceType distance_type=D_UNKNOWN;
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
	{
		CShiftInvariantKernel* shift_invariant_kernel=dynamic_cast<CShiftInvariantKernel*>(kernel_mgr.kernel_at(i));
		if (shift_invariant_kernel!=nullptr)
		{
			if (distance_type==D_UNKNOWN)
				distance_type=shift_invariant_kernel->get_distance_type();
			else if (distance_type==shift_invariant_kernel->get_distance_type())
				same=true;
			else
				break;
		}
		else
			break;
	}
	return same;
}
