/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <shogun/kernel/Kernel.h>
#include <shogun/statistical_testing/TwoSampleTest.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/TestTypes.h>

using namespace shogun;
using namespace internal;

struct CTwoSampleTest::Self
{
	Self(index_t num_kernels);
	KernelManager kernel_mgr;
};

CTwoSampleTest::Self::Self(index_t num_kernels) : kernel_mgr(num_kernels)
{
}

CTwoSampleTest::CTwoSampleTest() : CTwoDistributionTest()
{
	self=std::unique_ptr<Self>(new Self(TwoSampleTest::num_kernels));
}

CTwoSampleTest::CTwoSampleTest(CFeatures* samples_from_p, CFeatures* samples_from_q) : CTwoDistributionTest()
{
	self=std::unique_ptr<Self>(new Self(TwoSampleTest::num_kernels));
	set_p(samples_from_p);
	set_q(samples_from_q);
}

CTwoSampleTest::~CTwoSampleTest()
{
}

void CTwoSampleTest::set_kernel(CKernel* kernel)
{
	REQUIRE(kernel, "Kernel cannot be NULL!\n");
	self->kernel_mgr.kernel_at(0)=kernel;
	self->kernel_mgr.restore_kernel_at(0);
}

CKernel* CTwoSampleTest::get_kernel() const
{
	return get_kernel_mgr().kernel_at(0);
}

const char* CTwoSampleTest::get_name() const
{
	return "TwoSampleTest";
}

KernelManager& CTwoSampleTest::get_kernel_mgr()
{
	return self->kernel_mgr;
}

const KernelManager& CTwoSampleTest::get_kernel_mgr() const
{
	return self->kernel_mgr;
}
