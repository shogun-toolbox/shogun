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

#ifndef CROSS_VALIDATION_MMD_H_
#define CROSS_VALIDATION_MMD_H_

#include <memory>
#include <algorithm>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/statistical_testing/internals/mmd/PermutationMMD.h>

using std::unique_ptr;

namespace shogun
{

namespace internal
{

namespace mmd
{

struct CrossValidationMMD : PermutationMMD
{
	CrossValidationMMD(index_t n_x, index_t n_y, index_t num_folds, index_t num_null_samples)
	{
		ASSERT(n_x>0 && n_y>0);
		ASSERT(num_folds>0);
		ASSERT(num_null_samples>0);

		m_n_x=n_x;
		m_n_y=n_y;
		m_num_folds=num_folds;
		m_num_null_samples=num_null_samples;
		m_num_runs=DEFAULT_NUM_RUNS;
		m_alpha=DEFAULT_ALPHA;

		init();
	}

	void operator()(const KernelManager& kernel_mgr)
	{
		REQUIRE(m_rejections.num_rows==m_num_runs*m_num_folds,
			"Number of rows in the measure matrix (was %d), has to be >= %d*%d = %d!\n",
			m_rejections.num_rows, m_num_runs, m_num_folds, m_num_runs*m_num_folds);
		REQUIRE(size_t(m_rejections.num_cols)==kernel_mgr.num_kernels(),
			"Number of columns in the measure matrix (was %d), has to equal to the nunber of kernels (%d)!\n",
			m_rejections.num_cols, kernel_mgr.num_kernels());

		const index_t size=m_n_x+m_n_y;
		const index_t orig_n_x=m_n_x;
		const index_t orig_n_y=m_n_y;
		SGVector<float64_t> null_samples(m_num_null_samples);
		SGVector<float32_t> precomputed_km(size*(size+1)/2);

		for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
		{
			auto kernel=kernel_mgr.kernel_at(k);
			for (auto i=0; i<size; ++i)
			{
				for (auto j=i; j<size; ++j)
				{
					auto index=i*size-i*(i+1)/2+j;
					precomputed_km[index]=kernel->kernel(i, j);
				}
			}

			for (auto current_run=0; current_run<m_num_runs; ++current_run)
			{
				m_kfold_x->build_subsets();
				m_kfold_y->build_subsets();
				for (auto current_fold=0; current_fold<m_num_folds; ++current_fold)
				{
					generate_inds(current_fold);
					std::fill(m_inverted_inds.data(), m_inverted_inds.data()+m_inverted_inds.size(), -1);
					for (size_t idx=0; idx<m_xy_inds.size(); ++idx)
						m_inverted_inds[m_xy_inds[idx]]=idx;

					SGVector<index_t> xy_wrapper(m_xy_inds.data(), m_xy_inds.size(), false);
					m_stack->add_subset(xy_wrapper);

					m_permuted_inds.resize(m_xy_inds.size());
					SGVector<index_t> permutation_wrapper(m_permuted_inds.data(), m_permuted_inds.size(), false);
					for (auto n=0; n<m_num_null_samples; ++n)
					{
						std::iota(m_permuted_inds.data(), m_permuted_inds.data()+m_permuted_inds.size(), 0);
						CMath::permute(permutation_wrapper);

						m_stack->add_subset(permutation_wrapper);
						SGVector<index_t> inds=m_stack->get_last_subset()->get_subset_idx();
						m_stack->remove_subset();

						std::fill(m_inverted_permuted_inds[n].data(), m_inverted_permuted_inds[n].data()+size, -1);
						for (int idx=0; idx<inds.size(); ++idx)
							m_inverted_permuted_inds[n][inds[idx]]=idx;
					}
					m_stack->remove_subset();

					terms_t terms;
					for (auto i=0; i<size; ++i)
					{
						for (auto j=i; j<size; ++j)
						{
							auto inverted_row=m_inverted_inds[i];
							auto inverted_col=m_inverted_inds[j];
							if (inverted_row!=-1 && inverted_col!=-1)
							{
								auto idx=i*size-i*(i+1)/2+j;
								add_term_upper(terms, precomputed_km[idx], inverted_row, inverted_col);
							}
						}
					}
					auto statistic=compute(terms);

#pragma omp parallel for
					for (auto n=0; n<m_num_null_samples; ++n)
					{
						terms_t null_terms;
						for (auto i=0; i<size; ++i)
						{
							for (auto j=i; j<size; ++j)
							{
								auto inverted_row=m_inverted_permuted_inds[n][i];
								auto inverted_col=m_inverted_permuted_inds[n][j];
								if (inverted_row!=-1 && inverted_col!=-1)
								{
									auto idx=i*size-i*(i+1)/2+j;
									if (inverted_row<=inverted_col)
										add_term_upper(null_terms, precomputed_km[idx], inverted_row, inverted_col);
									else
										add_term_upper(null_terms, precomputed_km[idx], inverted_col, inverted_row);
								}
							}
						}
						null_samples[n]=compute(null_terms);
					}

					std::sort(null_samples.data(), null_samples.data()+null_samples.size());
					SG_SDEBUG("statistic=%f\n", statistic);
					float64_t idx=null_samples.find_position_to_insert(statistic);
					SG_SDEBUG("index=%f\n", idx);
					auto p_value=1.0-idx/m_num_null_samples;
					bool rejected=p_value<m_alpha;
					SG_SDEBUG("p-value=%f, alpha=%f, rejected=%d\n", p_value, m_alpha, rejected);
					m_rejections(current_run*m_num_folds+current_fold, k)=rejected;

					m_n_x=orig_n_x;
					m_n_y=orig_n_y;
				}
			}
		}
	}

	index_t m_num_runs;
	index_t m_num_folds;
	static constexpr index_t DEFAULT_NUM_RUNS=10;

	float64_t m_alpha;
	static constexpr float64_t DEFAULT_ALPHA=0.05;

	unique_ptr<CCrossValidationSplitting> m_kfold_x;
	unique_ptr<CCrossValidationSplitting> m_kfold_y;
	unique_ptr<CSubsetStack> m_stack;

	std::vector<index_t> m_xy_inds;
	SGVector<index_t> m_inverted_inds;
	SGMatrix<float64_t> m_rejections;

	void init()
	{
		SGVector<int64_t> dummy_labels_x(m_n_x);
		SGVector<int64_t> dummy_labels_y(m_n_y);

		auto instance_x=new CCrossValidationSplitting(new CBinaryLabels(dummy_labels_x), m_num_folds);
		auto instance_y=new CCrossValidationSplitting(new CBinaryLabels(dummy_labels_y), m_num_folds);
		m_kfold_x=unique_ptr<CCrossValidationSplitting>(instance_x);
		m_kfold_y=unique_ptr<CCrossValidationSplitting>(instance_y);

		m_stack=unique_ptr<CSubsetStack>(new CSubsetStack());

		const index_t size=m_n_x+m_n_y;
		m_inverted_inds=SGVector<index_t>(size);

		m_inverted_permuted_inds.resize(m_num_null_samples);
		for (auto i=0; i<m_num_null_samples; ++i)
			m_inverted_permuted_inds[i].resize(size);
	}

	void generate_inds(index_t current_fold)
	{
		SGVector<index_t> x_inds=m_kfold_x->generate_subset_inverse(current_fold);
		SGVector<index_t> y_inds=m_kfold_y->generate_subset_inverse(current_fold);
		std::for_each(y_inds.data(), y_inds.data()+y_inds.size(), [this](index_t& val) { val += m_n_x; });

		m_n_x=x_inds.size();
		m_n_y=y_inds.size();

		m_xy_inds.resize(x_inds.size()+y_inds.size());
		std::copy(x_inds.data(), x_inds.data()+x_inds.size(), m_xy_inds.data());
		std::copy(y_inds.data(), y_inds.data()+y_inds.size(), m_xy_inds.data()+x_inds.size());
	}
};

}

}

}
#endif // CROSS_VALIDATION_MMD_H_
