/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
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

#ifndef PERMUTATION_MMD_H_
#define PERMUTATION_MMD_H_

#include <algorithm>
#include <numeric>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/internals/mmd/ComputeMMD.h>

namespace shogun
{

namespace internal
{

namespace mmd
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct PermutationMMD : ComputeMMD
{
	PermutationMMD() : m_save_inds(false)
	{
	}

	template <class Kernel>
	SGVector<float32_t> operator()(const Kernel& kernel)
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		ASSERT(m_num_null_samples>0);
		precompute_permutation_inds();

		const index_t size=m_n_x+m_n_y;
		SGVector<float32_t> null_samples(m_num_null_samples);
#pragma omp parallel for
		for (auto n=0; n<m_num_null_samples; ++n)
		{
			terms_t terms;
			for (auto j=0; j<size; ++j)
			{
				auto inverted_col=m_inverted_permuted_inds(j, n);
				for (auto i=j; i<size; ++i)
				{
					auto inverted_row=m_inverted_permuted_inds(i, n);

					if (inverted_row>=inverted_col)
						add_term_lower(terms, kernel(i, j), inverted_row, inverted_col);
					else
						add_term_lower(terms, kernel(i, j), inverted_col, inverted_row);
				}
			}
			null_samples[n]=compute(terms);
			SG_SDEBUG("null_samples[%d] = %f!\n", n, null_samples[n]);
		}
		return null_samples;
	}

	SGMatrix<float32_t> operator()(const KernelManager& kernel_mgr)
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		ASSERT(m_num_null_samples>0);
		precompute_permutation_inds();

		const index_t size=m_n_x+m_n_y;
		SGMatrix<float32_t> null_samples(m_num_null_samples, kernel_mgr.num_kernels());
		SGVector<float32_t> km(size*(size+1)/2);
		for (auto k=0; k<kernel_mgr.num_kernels(); ++k)
		{
			auto kernel=kernel_mgr.kernel_at(k);
			terms_t terms;
			for (auto i=0; i<size; ++i)
			{
				for (auto j=i; j<size; ++j)
				{
					auto index=i*size-i*(i+1)/2+j;
					km[index]=kernel->kernel(i, j);
				}
			}

#pragma omp parallel for
			for (auto n=0; n<m_num_null_samples; ++n)
			{
				terms_t null_terms;
				for (auto i=0; i<size; ++i)
				{
					auto inverted_row=m_inverted_permuted_inds(i, n);
					auto index_base=i*size-i*(i+1)/2;
					for (auto j=i; j<size; ++j)
					{
						auto index=index_base+j;
						auto inverted_col=m_inverted_permuted_inds(j, n);

						if (inverted_row<=inverted_col)
							add_term_upper(null_terms, km[index], inverted_row, inverted_col);
						else
							add_term_upper(null_terms, km[index], inverted_col, inverted_row);
					}
				}
				null_samples(n, k)=compute(null_terms);
			}
		}
		return null_samples;
	}

	template <class Kernel>
	float64_t p_value(const Kernel& kernel)
	{
		auto statistic=ComputeMMD::operator()(kernel);
		auto null_samples=operator()(kernel);
		return compute_p_value(null_samples, statistic);
	}

	SGVector<float64_t> p_value(const KernelManager& kernel_mgr)
	{
		ASSERT(m_n_x>0 && m_n_y>0);
		ASSERT(m_num_null_samples>0);
		precompute_permutation_inds();

		const index_t size=m_n_x+m_n_y;
		SGVector<float32_t> null_samples(m_num_null_samples);
		SGVector<float64_t> result(kernel_mgr.num_kernels());

		SGVector<float32_t> km(size*(size+1)/2);
		for (auto k=0; k<kernel_mgr.num_kernels(); ++k)
		{
			auto kernel=kernel_mgr.kernel_at(k);
			terms_t terms;
			for (auto i=0; i<size; ++i)
			{
				for (auto j=i; j<size; ++j)
				{
					auto index=i*size-i*(i+1)/2+j;
					km[index]=kernel->kernel(i, j);
					add_term_upper(terms, km[index], i, j);
				}
			}
			float32_t statistic=compute(terms);
			SG_SDEBUG("Kernel(%d): statistic=%f\n", k, statistic);

#pragma omp parallel for
			for (auto n=0; n<m_num_null_samples; ++n)
			{
				terms_t null_terms;
				for (auto i=0; i<size; ++i)
				{
					auto inverted_row=m_inverted_permuted_inds(i, n);
					auto index_base=i*size-i*(i+1)/2;
					for (auto j=i; j<size; ++j)
					{
						auto index=index_base+j;
						auto inverted_col=m_inverted_permuted_inds(j, n);

						if (inverted_row<=inverted_col)
							add_term_upper(null_terms, km[index], inverted_row, inverted_col);
						else
							add_term_upper(null_terms, km[index], inverted_col, inverted_row);
					}
				}
				null_samples[n]=compute(null_terms);
			}
			result[k]=compute_p_value(null_samples, statistic);
			SG_SDEBUG("Kernel(%d): p_value=%f\n", k, result[k]);
		}

		return result;
	}

	inline void precompute_permutation_inds()
	{
		ASSERT(m_num_null_samples>0);
		allocate_permutation_inds();
		for (auto n=0; n<m_num_null_samples; ++n)
		{
			std::iota(m_permuted_inds.data(), m_permuted_inds.data()+m_permuted_inds.size(), 0);
			CMath::permute(m_permuted_inds);
			if (m_save_inds)
			{
				auto offset=n*m_permuted_inds.size();
				std::copy(m_permuted_inds.data(), m_permuted_inds.data()+m_permuted_inds.size(), &m_all_inds.matrix[offset]);
			}
			for (index_t i=0; i<m_permuted_inds.size(); ++i)
				m_inverted_permuted_inds(m_permuted_inds[i], n)=i;
		}
	}

	inline float64_t compute_p_value(SGVector<float32_t>& null_samples, float32_t statistic) const
	{
		std::sort(null_samples.data(), null_samples.data()+null_samples.size());
		float64_t idx=null_samples.find_position_to_insert(statistic);
		return 1.0-idx/null_samples.size();
	}

	inline void allocate_permutation_inds()
	{
		const index_t size=m_n_x+m_n_y;
		if (m_permuted_inds.size()!=size)
			m_permuted_inds=SGVector<index_t>(size);

		if (m_inverted_permuted_inds.num_cols!=m_num_null_samples || m_inverted_permuted_inds.num_rows!=size)
			m_inverted_permuted_inds=SGMatrix<index_t>(size, m_num_null_samples);

		if (m_save_inds && (m_all_inds.num_cols!=m_num_null_samples || m_all_inds.num_rows!=size))
			m_all_inds=SGMatrix<index_t>(size, m_num_null_samples);
	}

	index_t m_num_null_samples;
	bool m_save_inds;
	SGVector<index_t> m_permuted_inds;
	SGMatrix<index_t> m_inverted_permuted_inds;
	SGMatrix<index_t> m_all_inds;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS
}

}

}

#endif // PERMUTATION_MMD_H_
