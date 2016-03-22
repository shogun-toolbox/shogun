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

#ifndef KERNEL_MANAGER_H__
#define KERNEL_MANAGER_H__

#include <vector>
#include <memory>
#include <shogun/lib/common.h>
#include <shogun/statistical_testing/internals/InitPerKernel.h>

namespace shogun
{

class CKernel;
class CCustomKernel;

namespace internal
{

class KernelManager
{
public:
	KernelManager(index_t num_kernels);
	~KernelManager();

	InitPerKernel kernel_at(index_t i);
	CKernel* kernel_at(index_t i) const;

	void precompute_kernel_at(index_t i);
	void restore_kernel_at(index_t i);
private:
	std::vector<std::shared_ptr<CKernel>> m_kernels;
	std::vector<std::shared_ptr<CCustomKernel>> m_precomputed_kernels;
};

}

}

#endif // KERNEL_MANAGER_H__
