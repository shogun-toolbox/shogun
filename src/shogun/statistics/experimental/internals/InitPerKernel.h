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

#ifndef INIT_PER_KERNEL_H__
#define INIT_PER_KERNEL_H__

#include <memory>
#include <shogun/lib/common.h>

namespace shogun
{

class CKernel;

namespace internal
{

class InitPerKernel
{
	friend class KernelManager;
private:
	explicit InitPerKernel(std::shared_ptr<CKernel>& kernel);
public:
	~InitPerKernel();
	InitPerKernel& operator=(CKernel* kernel);
	operator CKernel*() const;
private:
	std::shared_ptr<CKernel>& m_kernel;
};

}

}

#endif // INIT_PER_KERNEL_H__
