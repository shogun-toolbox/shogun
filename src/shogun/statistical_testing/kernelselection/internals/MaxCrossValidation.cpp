/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Heiko Strathmann
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

#include <algorithm>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/mmd/CrossValidationMMD.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxCrossValidation.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

MaxCrossValidation::MaxCrossValidation(KernelManager& km, CMMD* est, const index_t& M, const index_t& K, const float64_t& alp)
: KernelSelection(km, est), num_runs(M), num_folds(K),  alpha(alp)
{
	REQUIRE(num_runs>0, "Number of runs (%d) must be positive!\n", num_runs);
	REQUIRE(num_folds>0, "Number of folds (%d) must be positive!\n", num_folds);
	REQUIRE(alpha>=0.0 && alpha<=1.0, "Threshold (%f) has to be in [0, 1]!\n", alpha);
}

MaxCrossValidation::~MaxCrossValidation()
{
}

SGVector<float64_t> MaxCrossValidation::get_measure_vector()
{
	return measures;
}

SGMatrix<float64_t> MaxCrossValidation::get_measure_matrix()
{
	return rejections;
}

void MaxCrossValidation::init_measures()
{
	const index_t num_kernels=kernel_mgr.num_kernels();
	if (rejections.num_rows!=num_folds*num_runs || rejections.num_cols!=num_kernels)
		rejections=SGMatrix<float64_t>(num_folds*num_runs, num_kernels);
	std::fill(rejections.data(), rejections.data()+rejections.size(), 0);
	if (measures.size()!=num_kernels)
		measures=SGVector<float64_t>(num_kernels);
	std::fill(measures.data(), measures.data()+measures.size(), 0);
}

void MaxCrossValidation::compute_measures()
{
	SG_SDEBUG("Performing %d fold cross-validattion!\n", num_folds);
	const size_t num_kernels=kernel_mgr.num_kernels();

	CQuadraticTimeMMD* quadratic_time_mmd=dynamic_cast<CQuadraticTimeMMD*>(estimator);
	if (quadratic_time_mmd)
	{
		REQUIRE(estimator->get_null_approximation_method()==NAM_PERMUTATION,
			"Only supported with PERMUTATION method for null distribution approximation!\n");

		auto Nx=estimator->get_num_samples_p();
		auto Ny=estimator->get_num_samples_q();
		auto num_null_samples=estimator->get_num_null_samples();
		auto stype=estimator->get_statistic_type();
		CrossValidationMMD compute(Nx, Ny, num_folds, num_null_samples);
		compute.m_stype=stype;
		compute.m_alpha=alpha;
		compute.m_num_runs=num_runs;
		compute.m_rejections=rejections;

		if (kernel_mgr.same_distance_type())
		{
			CDistance* distance=kernel_mgr.get_distance_instance();
			kernel_mgr.set_precomputed_distance(estimator->compute_joint_distance(distance));
			SG_UNREF(distance);
			compute(kernel_mgr);
			kernel_mgr.unset_precomputed_distance();
		}
		else
		{
			auto samples_p_and_q=quadratic_time_mmd->get_p_and_q();
			SG_REF(samples_p_and_q);

			for (size_t k=0; k<num_kernels; ++k)
			{
				CKernel* kernel=kernel_mgr.kernel_at(k);
				kernel->init(samples_p_and_q, samples_p_and_q);
			}

			compute(kernel_mgr);

			for (size_t k=0; k<num_kernels; ++k)
			{
				CKernel* kernel=kernel_mgr.kernel_at(k);
				kernel->remove_lhs_and_rhs();
			}

			SG_UNREF(samples_p_and_q);
		}
	}
	else // TODO put check, this one assumes infinite data
	{
		auto existing_kernel=estimator->get_kernel();
		for (auto i=0; i<num_runs; ++i)
		{
			for (auto j=0; j<num_folds; ++j)
			{
				SG_SDEBUG("Running fold %d\n", j);
				for (size_t k=0; k<num_kernels; ++k)
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

CKernel* MaxCrossValidation::select_kernel()
{
	init_measures();
	compute_measures();
	auto max_element=std::max_element(measures.vector, measures.vector+measures.vlen);
	auto max_idx=std::distance(measures.vector, max_element);
	SG_SDEBUG("Selected kernel at %d position!\n", max_idx);
	return kernel_mgr.kernel_at(max_idx);
}
