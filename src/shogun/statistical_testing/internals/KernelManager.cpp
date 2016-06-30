/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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
	SG_SDEBUG("Kernel manager instance initialized!\n");
}

KernelManager::KernelManager(index_t num_kernels)
{
	SG_SDEBUG("Kernel manager instance initialized with %d kernels!\n", num_kernels);
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

InitPerKernel KernelManager::kernel_at(size_t i)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<num_kernels(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, num_kernels()-1);
	SG_SDEBUG("Leaving!\n");
	return InitPerKernel(m_kernels[i]);
}

CKernel* KernelManager::kernel_at(size_t i) const
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<num_kernels(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, num_kernels()-1);
	if (m_precomputed_kernels[i]==nullptr)
	{
		SG_SDEBUG("Leaving!\n");
		return m_kernels[i].get();
	}
	SG_SDEBUG("Precomputed kernel exists!\n");
	SG_SDEBUG("Leaving!\n");
	return m_precomputed_kernels[i].get();
}

void KernelManager::push_back(CKernel* kernel)
{
	SG_SDEBUG("Entering!\n");
	SG_REF(kernel);
	m_kernels.push_back(std::shared_ptr<CKernel>(kernel, [](CKernel* ptr) { SG_UNREF(ptr); }));
	m_precomputed_kernels.push_back(nullptr);
	SG_SDEBUG("Leaving!\n");
}

const size_t KernelManager::num_kernels() const
{
	return m_kernels.size();
}

void KernelManager::precompute_kernel_at(size_t i)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<num_kernels(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, num_kernels()-1);
	auto kernel=m_kernels[i].get();
	if (kernel->get_kernel_type()!=K_CUSTOM)
	{
		// TODO give option to use different policies to precompute the kernel matrix
		// this one here is default setting : use shogun's pthread parallelism to compute
		// the kernel matrix.
		SGMatrix<float32_t> kernel_matrix=kernel->get_kernel_matrix<float32_t>();
		m_precomputed_kernels[i]=std::shared_ptr<CCustomKernel>(new CCustomKernel(kernel_matrix));
	}
	SG_SDEBUG("Leaving!\n");
}

void KernelManager::restore_kernel_at(size_t i)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<num_kernels(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, num_kernels()-1);
	m_precomputed_kernels[i]=nullptr;
	SG_SDEBUG("Leaving!\n");
}

bool KernelManager::same_distance_type() const
{
	bool same=false;
	EDistanceType distance_type=D_UNKNOWN;
	for (size_t i=0; i<num_kernels(); ++i)
	{
		CShiftInvariantKernel* shift_invariant_kernel=dynamic_cast<CShiftInvariantKernel*>(kernel_at(i));
		if (shift_invariant_kernel!=nullptr)
		{
			if (distance_type==D_UNKNOWN)
				distance_type=shift_invariant_kernel->get_distance_type();
			else if (distance_type==shift_invariant_kernel->get_distance_type())
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
			SG_SINFO("Kernel at location %d is not of CShiftInvariantKernel type (was of %s type)!\n",
				i, kernel_at(i)->get_name());
			break;
		}
	}
	return same;
}

CDistance* KernelManager::get_distance_instance() const
{
	REQUIRE(same_distance_type(), "Distance types for all the kernels are not the same!\n");

	CDistance* distance=nullptr;
	CShiftInvariantKernel* kernel_0=dynamic_cast<CShiftInvariantKernel*>(kernel_at(0));
	REQUIRE(kernel_0, "Kernel (%s) must be of CShiftInvariantKernel type!\n", kernel_at(0)->get_name());
	if (kernel_0->get_distance_type()==D_EUCLIDEAN)
	{
		auto euclidean_distance=new CEuclideanDistance();
		euclidean_distance->set_disable_sqrt(true);
		distance=euclidean_distance;
	}
	else if (kernel_0->get_distance_type()==D_MANHATTAN)
	{
		auto manhattan_distance=new CManhattanMetric();
		distance=manhattan_distance;
	}
	else
	{
		SG_SERROR("Unsupported distance type!\n");
	}
	return distance;
}

void KernelManager::set_precomputed_distance(CCustomDistance* distance) const
{
	for (size_t i=0; i<num_kernels(); ++i)
	{
		CKernel* kernel=kernel_at(i);
		CShiftInvariantKernel* shift_inv_kernel=dynamic_cast<CShiftInvariantKernel*>(kernel);
		REQUIRE(shift_inv_kernel!=nullptr, "Kernel instance (was %s) must be of CShiftInvarintKernel type!\n", kernel->get_name());
		shift_inv_kernel->m_precomputed_distance=distance;
		shift_inv_kernel->num_lhs=distance->get_num_vec_lhs();
		shift_inv_kernel->num_rhs=distance->get_num_vec_rhs();
	}
}

void KernelManager::unset_precomputed_distance() const
{
	for (size_t i=0; i<num_kernels(); ++i)
	{
		CKernel* kernel=kernel_at(i);
		CShiftInvariantKernel* shift_inv_kernel=dynamic_cast<CShiftInvariantKernel*>(kernel);
		REQUIRE(shift_inv_kernel!=nullptr, "Kernel instance (was %s) must be of CShiftInvarintKernel type!\n", kernel->get_name());
		shift_inv_kernel->m_precomputed_distance=nullptr;
		shift_inv_kernel->num_lhs=0;
		shift_inv_kernel->num_rhs=0;
	}
}
