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

#include <vector>
#include <memory>
#include <iostream>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/hypothesistest/internals/KernelManager.h>

using namespace shogun;
using namespace internal;

KernelManager::KernelManager(index_t num_kernels)
{
	m_kernels.resize(num_kernels);
	m_precomputed_kernels.resize(num_kernels);
	std::fill(m_kernels.begin(), m_kernels.end(), nullptr);
	std::fill(m_precomputed_kernels.begin(), m_precomputed_kernels.end(), nullptr);
}

KernelManager::~KernelManager()
{
}

InitPerKernel KernelManager::kernel_at(index_t i)
{
	std::cout << "KernelManager::kernel_at() : setting the kernel " << i << std::endl;
	ASSERT(i <= m_kernels.size());
	return InitPerKernel(m_kernels[i]);
}

CKernel* KernelManager::kernel_at(index_t i) const
{
	std::cout << "KernelManager::kernel_at() : getting the kernel " << i << std::endl;
	ASSERT(i <= m_kernels.size());
	if (m_precomputed_kernels[i] == nullptr)
	{
		return m_kernels[i].get();
	}
	return m_precomputed_kernels[i].get();
}

void KernelManager::precompute_kernel_at(index_t i)
{
	std::cout << "KernelManager::precompute_kernel_at() : precomputing the kernel " << i << std::endl;
	ASSERT(i <= m_kernels.size());
	auto kernel = m_kernels[i].get();
	if (kernel->get_kernel_type() != K_CUSTOM)
	{
		// TODO give option to use different policies to precompute the kernel matrix
		// this one here is default setting : use shogun's pthread parallelism to compute
		// the kernel matrix.
		m_precomputed_kernels[i] = std::shared_ptr<CCustomKernel>(new CCustomKernel(kernel));
	}
}

void KernelManager::restore_kernel_at(index_t i)
{
	std::cout << "KernelManager::precompute_kernel_at() : restoring the kernel " << i << std::endl;
	ASSERT(i <= m_kernels.size());
	m_precomputed_kernels[i] = nullptr;
}
