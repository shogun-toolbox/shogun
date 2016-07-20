/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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

#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>

#ifndef KERNEL_FUNCTOR_H__
#define KERNEL_FUNCTOR_H__

namespace shogun
{

class CKernel;

namespace internal
{

class Kernel
{
public:
	explicit Kernel(CKernel* kernel) : m_kernel(kernel)
	{
	}

	inline float32_t operator()(int32_t i, int32_t j) const
	{
		return m_kernel->kernel(i, j);
	}
private:
	CKernel* m_kernel;
};

class SelfAdjointPrecomputedKernel
{
public:
	SelfAdjointPrecomputedKernel() : m_num_feat_vec(0)
	{
	}
	explicit SelfAdjointPrecomputedKernel(SGVector<float32_t> self_adjoint_kernel_matrix) : m_num_feat_vec(0)
	{
		REQUIRE(self_adjoint_kernel_matrix.size()>0, "Provided kernel matrix cannot be of size 0!\n");
		m_self_adjoint_kernel_matrix=self_adjoint_kernel_matrix;
	}
	void precompute(CKernel* kernel)
	{
		REQUIRE(kernel, "Kernel instance cannot be NULL!\n");
		REQUIRE(kernel->get_num_vec_lhs()==kernel->get_num_vec_rhs(),
			"Kernel instance is not symmetric (%dx%d)!\n", kernel->get_num_vec_lhs(), kernel->get_num_vec_rhs());
		m_num_feat_vec=kernel->get_num_vec_lhs();
		auto size=m_num_feat_vec*(m_num_feat_vec+1)/2;
		if (m_self_adjoint_kernel_matrix.size()==0 || m_self_adjoint_kernel_matrix.size()!=size)
			m_self_adjoint_kernel_matrix=SGVector<float32_t>(size);
		for (auto i=0; i<m_num_feat_vec; ++i)
		{
			for (auto j=i; j<m_num_feat_vec; ++j)
			{
				auto index=i*m_num_feat_vec-i*(i+1)/2+j;
				m_self_adjoint_kernel_matrix[index]=kernel->kernel(i, j);
			}
		}
	}
	inline float32_t operator()(int32_t i, int32_t j) const
	{
		ASSERT(m_num_feat_vec);
		ASSERT(i>=0 && i<m_num_feat_vec);
		ASSERT(j>=0 && j<m_num_feat_vec);
		if (i>j)
			std::swap(i, j);
		auto index=i*m_num_feat_vec-i*(i+1)/2+j;
		return m_self_adjoint_kernel_matrix[index];
	}
private:
	SGVector<float32_t> m_self_adjoint_kernel_matrix;
	index_t m_num_feat_vec;
};

}

}
#endif // KERNEL_FUNCTOR_H__
