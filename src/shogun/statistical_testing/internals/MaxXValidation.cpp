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
	SG_SNOTIMPLEMENTED;
	return SGVector<float64_t>();
}

SGMatrix<float64_t> MaxXValidation::get_measure_matrix()
{
	SG_SNOTIMPLEMENTED;
	return SGMatrix<float64_t>();
}

void MaxXValidation::compute_measures(SGVector<float64_t>& measures, SGVector<index_t>& term_counters)
{
	const size_t num_kernels=kernel_mgr.num_kernels();
	for (size_t i=0; i<num_kernels; ++i)
	{
		auto kernel=kernel_mgr.kernel_at(i);
		estimator->set_kernel(kernel);
		bool rejected=estimator->compute_p_value(estimator->compute_statistic())<alpha;
		auto delta=measures[i]-rejected;
		measures[i]=delta/term_counters[i]++;
		estimator->cleanup();
	}
}

CKernel* MaxXValidation::select_kernel()
{
	auto& dm=estimator->get_data_manager();
	dm.set_xvalidation_mode(true);
	auto existing_kernel=estimator->get_kernel();

	const index_t N=dm.get_num_folds();
	// TODO write a more meaningful error message
	REQUIRE(N!=0, "Number of folds is not set!\n");
	SG_SINFO("Performing %d fold cross-validattion!\n", N);
	// train mode is already ON by now! set by the caller
	SGVector<float64_t> measures(kernel_mgr.num_kernels());
	std::fill(measures.data(), measures.data()+measures.size(), 0);
	SGVector<index_t> term_counters(measures.size());
	std::fill(term_counters.data(), term_counters.data()+term_counters.size(), 1);
	for (auto i=0; i<num_run; ++i)
	{
		for (auto j=0; j<N; ++j)
		{
			dm.use_fold(j);
			compute_measures(measures, term_counters);
		}
	}

	estimator->set_kernel(existing_kernel);
	dm.set_xvalidation_mode(false);

	auto min_element=std::min_element(measures.vector, measures.vector+measures.vlen);
	auto min_idx=std::distance(measures.vector, min_element);
	SG_SDEBUG("Selected kernel at %d position!\n", min_idx);
	return kernel_mgr.kernel_at(min_idx);
}
