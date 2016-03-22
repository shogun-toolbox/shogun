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
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
 */

#include <shogun/kernel/Kernel.h>
#include <shogun/statistical_testing/internals/InitPerKernel.h>

using namespace shogun;
using namespace internal;

InitPerKernel::InitPerKernel(std::shared_ptr<CKernel>& kernel) : m_kernel(kernel)
{
}

InitPerKernel::~InitPerKernel()
{
}

InitPerKernel& InitPerKernel::operator=(CKernel* kernel)
{
	SG_REF(kernel);
	m_kernel = std::shared_ptr<CKernel>(kernel, [](CKernel* ptr) { SG_UNREF(ptr); });
	return *this;
}

InitPerKernel::operator CKernel*() const
{
	return m_kernel.get();
}
