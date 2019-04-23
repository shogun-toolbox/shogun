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

#include <algorithm>
#include <numeric>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/mmd/CrossValidationMMD.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxCrossValidation.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

template <typename PRNG>
MaxCrossValidation<PRNG>::MaxCrossValidation(
    KernelManager& km, std::shared_ptr<MMD> est, const index_t& M, const index_t& K,
    const float64_t& alp, PRNG& _prng)
    : KernelSelection(km, est), num_runs(M), num_folds(K), alpha(alp),
      prng(_prng)
{
	require(num_runs>0, "Number of runs ({}) must be positive!", num_runs);
	require(num_folds>0, "Number of folds ({}) must be positive!", num_folds);
	require(alpha>=0.0 && alpha<=1.0, "Threshold ({}) has to be in [0, 1]!", alpha);
}

template <typename PRNG>
MaxCrossValidation<PRNG>::~MaxCrossValidation()
{
}

template <typename PRNG>
SGVector<float64_t> MaxCrossValidation<PRNG>::get_measure_vector()
{
	return measures;
}

template <typename PRNG>
SGMatrix<float64_t> MaxCrossValidation<PRNG>::get_measure_matrix()
{
	return rejections;
}

template <typename PRNG>
void MaxCrossValidation<PRNG>::init_measures()
{
	const index_t num_kernels=kernel_mgr.num_kernels();
	if (rejections.num_rows!=num_folds*num_runs || rejections.num_cols!=num_kernels)
		rejections=SGMatrix<float64_t>(num_folds*num_runs, num_kernels);
	std::fill(rejections.data(), rejections.data()+rejections.size(), 0);
	if (measures.size()!=num_kernels)
		measures=SGVector<float64_t>(num_kernels);
	std::fill(measures.data(), measures.data()+measures.size(), 0);
}

template <typename PRNG>
void MaxCrossValidation<PRNG>::compute_measures()
{
	SG_DEBUG("Performing {} fold cross-validattion!", num_folds);
	const auto num_kernels=kernel_mgr.num_kernels();

	auto quadratic_time_mmd=std::dynamic_pointer_cast<QuadraticTimeMMD>(estimator);
	if (quadratic_time_mmd)
	{
		require(estimator->get_null_approximation_method()==NAM_PERMUTATION,
			"Only supported with PERMUTATION method for null distribution approximation!");

		auto Nx=estimator->get_num_samples_p();
		auto Ny=estimator->get_num_samples_q();
		auto num_null_samples=estimator->get_num_null_samples();
		auto stype=estimator->get_statistic_type();
		CrossValidationMMD compute(Nx, Ny, num_folds, num_null_samples, prng);
		compute.m_stype=stype;
		compute.m_alpha=alpha;
		compute.m_num_runs=num_runs;
		compute.m_rejections=rejections;

		if (kernel_mgr.same_distance_type())
		{
			std::shared_ptr<Distance> distance=kernel_mgr.get_distance_instance();
			auto precomputed_distance=estimator->compute_joint_distance(distance);
			kernel_mgr.set_precomputed_distance(precomputed_distance);
			compute(kernel_mgr, prng);
			kernel_mgr.unset_precomputed_distance();

		}
		else
		{
			auto samples_p_and_q=quadratic_time_mmd->get_p_and_q();


			for (auto k=0; k<num_kernels; ++k)
			{
				std::shared_ptr<Kernel> kernel=kernel_mgr.kernel_at(k);
				kernel->init(samples_p_and_q, samples_p_and_q);
			}

			compute(kernel_mgr, prng);

			for (auto k=0; k<num_kernels; ++k)
			{
				std::shared_ptr<Kernel> kernel=kernel_mgr.kernel_at(k);
				kernel->remove_lhs_and_rhs();
			}


		}
	}
	else // TODO put check, this one assumes infinite data
	{
		auto existing_kernel=estimator->get_kernel();
		for (auto i=0; i<num_runs; ++i)
		{
			for (auto j=0; j<num_folds; ++j)
			{
				SG_DEBUG("Running fold {}", j);
				for (auto k=0; k<num_kernels; ++k)
				{
					auto kernel=kernel_mgr.kernel_at(k);
					estimator->set_kernel(kernel);
					auto statistic=estimator->compute_statistic();
					rejections(i*num_folds+j, k)=estimator->compute_p_value(statistic)<alpha;
					estimator->cleanup();
				}
			}
		}
		if (existing_kernel)
			estimator->set_kernel(existing_kernel);
	}

	for (auto j=0; j<rejections.num_cols; ++j)
	{
		auto begin=rejections.get_column_vector(j);
		auto size=rejections.num_rows;
		measures[j]=std::accumulate(begin, begin+size, 0.0)/size;
	}
}

template <typename PRNG>
std::shared_ptr<Kernel> MaxCrossValidation<PRNG>::select_kernel()
{
	init_measures();
	compute_measures();
	auto max_element=std::max_element(measures.vector, measures.vector+measures.vlen);
	auto max_idx=std::distance(measures.vector, max_element);
	SG_DEBUG("Selected kernel at {} position!", max_idx);
	return kernel_mgr.kernel_at(max_idx);
}


template class shogun::internal::MaxCrossValidation<std::mt19937_64>;
