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
#include <shogun/statistical_testing/IndependenceTest.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/TestTypes.h>

using namespace shogun;
using namespace internal;

CIndependenceTest::CIndependenceTest() : CTwoDistributionTest(IndependenceTest::num_kernels)
{
}

CIndependenceTest::~CIndependenceTest()
{
}

void CIndependenceTest::set_kernel_p(CKernel* kernel_p)
{
	auto& km = get_kernel_manager();
	km.kernel_at(0) = kernel_p;
}

CKernel* CIndependenceTest::get_kernel_p() const
{
	const auto& km = get_kernel_manager();
	return km.kernel_at(0);
}

void CIndependenceTest::set_kernel_q(CKernel* kernel_q)
{
	auto& km = get_kernel_manager();
	km.kernel_at(1) = kernel_q;
}

CKernel* CIndependenceTest::get_kernel_q() const
{
	const auto& km = get_kernel_manager();
	return km.kernel_at(1);
}

const char* CIndependenceTest::get_name() const
{
	return "IndependenceTest";
}
