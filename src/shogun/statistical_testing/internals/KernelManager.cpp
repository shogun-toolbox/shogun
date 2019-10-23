/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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

#include <vector>
#include <memory>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/distance/ManhattanMetric.h>
#include <shogun/kernel/ShiftInvariantKernel.h>
#include <shogun/statistical_testing/internals/KernelManager.h>

using namespace shogun;
using namespace internal;

KernelManager::KernelManager()
{
	SG_TRACE("Kernel manager instance initialized!");
}

KernelManager::KernelManager(index_t num_kernels)
{
	SG_DEBUG("Kernel manager instance initialized with {} kernels!", num_kernels);
	m_kernels.resize(num_kernels);
	m_precomputed_kernels.resize(num_kernels);
	std::fill(m_kernels.begin(), m_kernels.end(), nullptr);
	std::fill(m_precomputed_kernels.begin(), m_precomputed_kernels.end(), nullptr);
}

KernelManager::~KernelManager()
{
	clear();
}

void KernelManager::clear()
{
	m_kernels.resize(0);
	m_precomputed_kernels.resize(0);
}

InitPerKernel KernelManager::kernel_at(index_t i)
{
	SG_TRACE("Entering!");
	require(i<num_kernels(),
			"Value of i ({}) should be between 0 and {}, inclusive!",
			i, num_kernels()-1);
	SG_TRACE("Leaving!");
	return InitPerKernel(m_kernels[i]);
}

std::shared_ptr<Kernel> KernelManager::kernel_at(index_t i) const
{
	SG_TRACE("Entering!");
	require(i<num_kernels(),
			"Value of i ({}) should be between 0 and {}, inclusive!",
			i, num_kernels()-1);
	if (m_precomputed_kernels[i]==nullptr)
	{
		SG_TRACE("Leaving!");
		return m_kernels[i];
	}
	SG_DEBUG("Precomputed kernel exists!");
	SG_TRACE("Leaving!");
	return m_precomputed_kernels[i];
}

void KernelManager::push_back(const std::shared_ptr<Kernel>& kernel)
{
	SG_TRACE("Entering!");
	m_kernels.push_back(kernel);
	m_precomputed_kernels.push_back(nullptr);
	SG_TRACE("Leaving!");
}

const index_t KernelManager::num_kernels() const
{
	// TODO in case there is an underflow, at least it is not silent
	// a better handling is to use index_t based Shogun data structures
	ASSERT((index_t)m_kernels.size()>=0);
	return (index_t)m_kernels.size();
}

void KernelManager::precompute_kernel_at(index_t i)
{
	SG_TRACE("Entering!");
	require(i<num_kernels(),
			"Value of i ({}) should be between 0 and {}, inclusive!",
			i, num_kernels()-1);
	auto kernel=m_kernels[i];
	if (kernel->get_kernel_type()!=K_CUSTOM)
	{
		// TODO give option to use different policies to precompute the kernel matrix
		// this one here is default setting : use shogun's pthread parallelism to compute
		// the kernel matrix.
		SGMatrix<float32_t> kernel_matrix=kernel->get_kernel_matrix<float32_t>();
		m_precomputed_kernels[i]=std::make_shared<CustomKernel>(kernel_matrix);
		SG_DEBUG("Kernel type {} is precomputed and replaced internally with {}!",
			kernel->get_name(), m_precomputed_kernels[i]->get_name());
	}
	SG_TRACE("Leaving!");
}

void KernelManager::restore_kernel_at(index_t i)
{
	SG_TRACE("Entering!");
	require(i<num_kernels(),
			"Value of i ({}) should be between 0 and {}, inclusive!",
			i, num_kernels()-1);
	m_precomputed_kernels[i]=nullptr;
	SG_DEBUG("Precomputed kernel (if any) was deleted!");
	SG_TRACE("Leaving!");
}

bool KernelManager::same_distance_type() const
{
	ASSERT(num_kernels()>0);
	bool same=false;
	EDistanceType distance_type=D_UNKNOWN;
	for (auto i=0; i<num_kernels(); ++i)
	{
		auto shift_invariant_kernel=kernel_at(i)->as<ShiftInvariantKernel>();
		if (shift_invariant_kernel!=nullptr)
		{
			auto current_distance_type=shift_invariant_kernel->get_distance_type();
			if (distance_type==D_UNKNOWN)
			{
				distance_type=current_distance_type;
				same=true;
			}
			else if (distance_type==current_distance_type)
				same=true;
			else
			{
				same=false;
				break;
			}
		}
		else
		{
			same=false;
			io::info("Kernel at location {} is not of CShiftInvariantKernel type (was of {} type)!",
				i, kernel_at(i)->get_name());
			break;
		}
	}
	return same;
}

std::shared_ptr<Distance> KernelManager::get_distance_instance() const
{
	require(same_distance_type(), "Distance types for all the kernels are not the same!");

	std::shared_ptr<Distance> distance=nullptr;
	auto kernel_0=kernel_at(0)->as<ShiftInvariantKernel>();
	require(kernel_0, "Kernel ({}) must be of CShiftInvariantKernel type!", kernel_at(0)->get_name());
	if (kernel_0->get_distance_type()==D_EUCLIDEAN)
	{
		auto euclidean_distance=std::make_shared<EuclideanDistance>();
		euclidean_distance->set_disable_sqrt(true);
		distance=euclidean_distance;
	}
	else if (kernel_0->get_distance_type()==D_MANHATTAN)
	{
		auto manhattan_distance=std::make_shared<ManhattanMetric>();
		distance=manhattan_distance;
	}
	else
	{
		error("Unsupported distance type!");
	}
	return distance;
}

void KernelManager::set_precomputed_distance(const std::shared_ptr<CustomDistance>& distance) const
{
	require(distance!=nullptr, "Distance instance cannot be null!");
	for (auto i=0; i<num_kernels(); ++i)
	{
		std::shared_ptr<Kernel> kernel=kernel_at(i);
		auto shift_inv_kernel=kernel->as<ShiftInvariantKernel>();
		require(shift_inv_kernel!=nullptr, "Kernel instance (was {}) must be of CShiftInvarintKernel type!", kernel->get_name());
		shift_inv_kernel->m_precomputed_distance=distance;
		shift_inv_kernel->num_lhs=distance->get_num_vec_lhs();
		shift_inv_kernel->num_rhs=distance->get_num_vec_rhs();
	}
}

void KernelManager::unset_precomputed_distance() const
{
	for (auto i=0; i<num_kernels(); ++i)
	{
		std::shared_ptr<Kernel> kernel=kernel_at(i);
		auto shift_inv_kernel=kernel->as<ShiftInvariantKernel>();
		require(shift_inv_kernel!=nullptr, "Kernel instance (was {}) must be of CShiftInvarintKernel type!", kernel->get_name());
		shift_inv_kernel->m_precomputed_distance=nullptr;
		shift_inv_kernel->num_lhs=0;
		shift_inv_kernel->num_rhs=0;
	}
}
