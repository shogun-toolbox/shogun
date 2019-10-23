/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2017 Soumyajit De
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

#include <shogun/kernel/Kernel.h>
#include <shogun/statistical_testing/IndependenceTest.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/TestTypes.h>

#include <utility>

using namespace shogun;

struct IndependenceTest::Self
{
	Self(index_t num_kernels);
	internal::KernelManager kernel_mgr;
};

IndependenceTest::Self::Self(index_t num_kernels) : kernel_mgr(num_kernels)
{
}

IndependenceTest::IndependenceTest() : TwoDistributionTest()
{
	self=std::unique_ptr<Self>(new Self(internal::IndependenceTest::num_kernels));
}

IndependenceTest::~IndependenceTest()
{
}

void IndependenceTest::set_kernel_p(std::shared_ptr<Kernel> kernel_p)
{
	self->kernel_mgr.kernel_at(0)=std::move(kernel_p);
}

std::shared_ptr<Kernel> IndependenceTest::get_kernel_p() const
{
	return self->kernel_mgr.kernel_at(0);
}

void IndependenceTest::set_kernel_q(std::shared_ptr<Kernel> kernel_q)
{
	self->kernel_mgr.kernel_at(1)=std::move(kernel_q);
}

std::shared_ptr<Kernel> IndependenceTest::get_kernel_q() const
{
	return self->kernel_mgr.kernel_at(1);
}

const char* IndependenceTest::get_name() const
{
	return "IndependenceTest";
}

internal::KernelManager& IndependenceTest::get_kernel_mgr()
{
	return self->kernel_mgr;
}

const internal::KernelManager& IndependenceTest::get_kernel_mgr() const
{
	return self->kernel_mgr;
}
