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

#ifndef COMPUTE_MMD_H_
#define COMPUTE_MMD_H_

#include <array>
#include <vector>
#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

namespace internal
{

namespace mmd
{

struct terms_t
{
	std::array<float64_t, 3> term{};
	std::array<float64_t, 3> diag{};
};

/**
 * @brief Class Compute blah blah.
 */
struct ComputeMMD
{
	ComputeMMD() : m_n_x(0), m_n_y(0), m_stype(EStatisticType::UNBIASED_FULL)
	{
	}

	template <class Kernel>
	float32_t operator()(const Kernel& kernel) const
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		const index_t size=m_n_x+m_n_y;
		terms_t terms;
		for (auto i=0; i<size; ++i)
		{
			for (auto j=i; j<size; ++j)
				add_term_upper(terms, kernel(i, j), i, j);
		}
		return compute(terms);
	}

	template <typename T>
	float32_t operator()(const SGMatrix<T>& kernel_matrix) const
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		const index_t size=m_n_x+m_n_y;
		ASSERT(kernel_matrix.num_rows==size && kernel_matrix.num_cols==size);

		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
		typedef Eigen::Block<Eigen::Map<const MatrixXt> > BlockXt;

		Eigen::Map<const MatrixXt> map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);

		const BlockXt& b_x=map.block(0, 0, m_n_x, m_n_x);
		const BlockXt& b_y=map.block(m_n_x, m_n_x, m_n_y, m_n_y);
		const BlockXt& b_xy=map.block(m_n_x, 0, m_n_y, m_n_x);

		terms_t terms;
		terms.diag[0]=b_x.diagonal().sum();
		terms.diag[1]=b_y.diagonal().sum();
		terms.diag[2]=b_xy.diagonal().sum();

		terms.term[0]=(b_x.sum()-terms.diag[0])/2+terms.diag[0];
		terms.term[1]=(b_y.sum()-terms.diag[1])/2+terms.diag[1];
		terms.term[2]=b_xy.sum();

		return compute(terms);
	}

	SGVector<float64_t> operator()(const KernelManager& kernel_mgr) const
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		std::vector<terms_t> terms(kernel_mgr.num_kernels());
		const index_t size=m_n_x+m_n_y;
		for (auto j=0; j<size; ++j)
		{
			for (auto i=j; i<size; ++i)
			{
				for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
				{
					auto kernel=kernel_mgr.kernel_at(k)->kernel(i, j);
					add_term_lower(terms[k], kernel, i, j);
				}
			}
		}

		SGVector<float64_t> result(kernel_mgr.num_kernels());
		for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
		{
			result[k]=compute(terms[k]);
			SG_SDEBUG("result[%d] = %f!\n", k, result[k]);
		}
		terms.resize(0);
		return result;
	}

	/**
	 * Adds the kernel value to to the term that corresponding to K(i, j). It only
	 * uses the lower triangular half of the matrix to exploit symmetry.
	 *
	 * @param terms the terms for computing MMD
	 * @param kernel_value the kernel value between i-th and j-th features.
	 * @param i the row index for the Gram matrix
	 * @param j the col index for the Gram matrix
	 */
	template <typename T>
	inline void add_term_lower(terms_t& terms, T kernel_value, index_t i, index_t j) const
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		if (i<m_n_x && j<m_n_x && i>=j)
		{
			SG_SDEBUG("Adding Kernel(%d, %d)=%f to term_0!\n", i, j, kernel_value);
			terms.term[0]+=kernel_value;
			if (i==j)
				terms.diag[0]+=kernel_value;
		}
		else if (i>=m_n_x && j>=m_n_x && i>=j)
		{
			SG_SDEBUG("Adding Kernel(%d, %d)=%f to term_1!\n", i, j, kernel_value);
			terms.term[1]+=kernel_value;
			if (i==j)
				terms.diag[1]+=kernel_value;
		}
		else if (i>=m_n_x && j<m_n_x)
		{
			SG_SDEBUG("Adding Kernel(%d, %d)=%f to term_2!\n", i, j, kernel_value);
			terms.term[2]+=kernel_value;
			if (i-m_n_x==j)
				terms.diag[2]+=kernel_value;
		}
	}

	/**
	 * Adds the kernel value to to the term that corresponding to K(i, j). It only
	 * uses the upper triangular half of the matrix to exploit symmetry.
	 *
	 * @param terms the terms for computing MMD
	 * @param kernel_value the kernel value between i-th and j-th features.
	 * @param i the row index for the Gram matrix
	 * @param j the col index for the Gram matrix
	 */
	template <typename T>
	inline void add_term_upper(terms_t& terms, T kernel_value, index_t i, index_t j) const
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		if (i<m_n_x && j<m_n_x && i<=j)
		{
			SG_SDEBUG("Adding Kernel(%d, %d)=%f to term_0!\n", i, j, kernel_value);
			terms.term[0]+=kernel_value;
			if (i==j)
				terms.diag[0]+=kernel_value;
		}
		else if (i>=m_n_x && j>=m_n_x && i<=j)
		{
			SG_SDEBUG("Adding Kernel(%d, %d)=%f to term_1!\n", i, j, kernel_value);
			terms.term[1]+=kernel_value;
			if (i==j)
				terms.diag[1]+=kernel_value;
		}
		else if (i<m_n_x && j>=m_n_x)
		{
			SG_SDEBUG("Adding Kernel(%d, %d)=%f to term_2!\n", i, j, kernel_value);
			terms.term[2]+=kernel_value;
			if (i+m_n_x==j)
				terms.diag[2]+=kernel_value;
		}
	}

	inline float64_t compute(terms_t& terms) const
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		terms.term[0]=2*(terms.term[0]-terms.diag[0]);
		terms.term[1]=2*(terms.term[1]-terms.diag[1]);
		SG_SDEBUG("term_0 sum (without diagonal) = %f!\n", terms.term[0]);
		SG_SDEBUG("term_1 sum (without diagonal) = %f!\n", terms.term[1]);
		if (m_stype!=EStatisticType::BIASED_FULL)
		{
			terms.term[0]/=m_n_x*(m_n_x-1);
			terms.term[1]/=m_n_y*(m_n_y-1);
		}
		else
		{
			terms.term[0]+=terms.diag[0];
			terms.term[1]+=terms.diag[1];
			SG_SDEBUG("term_0 sum (with diagonal) = %f!\n", terms.term[0]);
			SG_SDEBUG("term_1 sum (with diagonal) = %f!\n", terms.term[1]);
			terms.term[0]/=m_n_x*m_n_x;
			terms.term[1]/=m_n_y*m_n_y;
		}
		SG_SDEBUG("term_0 (normalized) = %f!\n", terms.term[0]);
		SG_SDEBUG("term_1 (normalized) = %f!\n", terms.term[1]);

		SG_SDEBUG("term_2 sum (with diagonal) = %f!\n", terms.term[2]);
		if (m_stype==EStatisticType::UNBIASED_INCOMPLETE)
		{
			terms.term[2]-=terms.diag[2];
			SG_SDEBUG("term_2 sum (without diagonal) = %f!\n", terms.term[2]);
			terms.term[2]/=m_n_x*(m_n_x-1);
		}
		else
			terms.term[2]/=m_n_x*m_n_y;
		SG_SDEBUG("term_2 (normalized) = %f!\n", terms.term[2]);

		auto result=terms.term[0]+terms.term[1]-2*terms.term[2];
		SG_SDEBUG("result = %f!\n", result);
		return result;
	}

	index_t m_n_x;
	index_t m_n_y;
	EStatisticType m_stype;
};

}

}

}
#endif // COMPUTE_MMD_H_
