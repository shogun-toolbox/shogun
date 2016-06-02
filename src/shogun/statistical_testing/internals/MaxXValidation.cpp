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
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/MaxXValidation.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/DataManager.h>

using namespace shogun;
using namespace internal;

MaxXValidation::MaxXValidation(KernelManager& km, CMMD* est, const index_t& M, const float64_t& alp)
: KernelSelection(km, est), num_run(M), alpha(alp)
{
	REQUIRE(num_run>0, "Number of runs is %d!\n", num_run);
	REQUIRE(alpha>=0.0 && alpha<=1.0, "Threshold is %f!\n", alpha);
}

MaxXValidation::~MaxXValidation()
{
}

SGVector<float64_t> MaxXValidation::get_measure_vector()
{
	return measures;
}

SGMatrix<float64_t> MaxXValidation::get_measure_matrix()
{
	return rejections;
}

void MaxXValidation::init_measures()
{
	const index_t num_kernels=kernel_mgr.num_kernels();
	auto& data_mgr=estimator->get_data_mgr();
	const index_t N=data_mgr.get_num_folds();
	REQUIRE(N!=0, "Number of folds is not set!\n");
	if (rejections.num_rows!=N*num_run || rejections.num_cols!=num_kernels)
		rejections=SGMatrix<float64_t>(N*num_run, num_kernels);
	std::fill(rejections.data(), rejections.data()+rejections.size(), 0);
	if (measures.size()!=num_kernels)
		measures=SGVector<float64_t>(num_kernels);
	std::fill(measures.data(), measures.data()+measures.size(), 0);
}

void MaxXValidation::compute_measures()
{
	auto& data_mgr=estimator->get_data_mgr();
	data_mgr.set_cross_validation_mode(true);

	const index_t N=data_mgr.get_num_folds();
	SG_SINFO("Performing %d fold cross-validattion!\n", N);

	const size_t num_kernels=kernel_mgr.num_kernels();
	auto existing_kernel=estimator->get_kernel();
	for (auto i=0; i<num_run; ++i)
	{
		data_mgr.shuffle_features();
		for (auto j=0; j<N; ++j)
		{
			data_mgr.use_fold(j);
			SG_SDEBUG("Running fold %d\n", j);
			for (size_t k=0; k<num_kernels; ++k)
			{
				auto kernel=kernel_mgr.kernel_at(k);
				estimator->set_kernel(kernel);
				auto statistic=estimator->compute_statistic();
				rejections(i*N+j, k)=estimator->compute_p_value(statistic)<alpha;
				estimator->cleanup();
			}
		}
		data_mgr.unshuffle_features();
	}
	data_mgr.set_cross_validation_mode(false);
	estimator->set_kernel(existing_kernel);

	for (auto j=0; j<rejections.num_cols; ++j)
	{
		auto begin=rejections.get_column_vector(j);
		auto size=rejections.num_rows;
		measures[j]=std::accumulate(begin, begin+size, 0)/size;
	}
}

CKernel* MaxXValidation::select_kernel()
{
	init_measures();
	compute_measures();
	auto max_element=std::max_element(measures.vector, measures.vector+measures.vlen);
	auto max_idx=std::distance(measures.vector, max_element);
	SG_SDEBUG("Selected kernel at %d position!\n", max_idx);
	return kernel_mgr.kernel_at(max_idx);
}
