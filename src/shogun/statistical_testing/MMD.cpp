/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
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

#include <shogun/kernel/Kernel.h>
#include <shogun/features/Features.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace internal;
using std::unique_ptr;
using std::shared_ptr;

struct CMMD::Self
{
	Self()
	{
		num_null_samples = DEFAULT_NUM_NULL_SAMPLES;
		stype = DEFAULT_STYPE;
		null_approximation_method = DEFAULT_NULL_APPROXIMATION_METHOD;
		strategy=unique_ptr<CKernelSelectionStrategy>(new CKernelSelectionStrategy());
	}

	index_t num_null_samples;
	EStatisticType stype;
	ENullApproximationMethod null_approximation_method;
	std::unique_ptr<CKernelSelectionStrategy> strategy;

	static constexpr index_t DEFAULT_NUM_NULL_SAMPLES = 250;
	static constexpr EStatisticType DEFAULT_STYPE = EStatisticType::UNBIASED_FULL;
	static constexpr ENullApproximationMethod DEFAULT_NULL_APPROXIMATION_METHOD = ENullApproximationMethod::PERMUTATION;
};

CMMD::CMMD() : CTwoSampleTest()
{
	init();
}

CMMD::CMMD(CFeatures* samples_from_p, CFeatures* samples_from_q) : CTwoSampleTest(samples_from_p, samples_from_q)
{
	init();
}

void CMMD::init()
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	self=unique_ptr<Self>(new Self());
}

CMMD::~CMMD()
{
	cleanup();
}

void CMMD::set_kernel_selection_strategy(EKernelSelectionMethod method, bool weighted)
{
	self->strategy->use_method(method)
		.use_weighted(weighted);
}

void CMMD::set_kernel_selection_strategy(EKernelSelectionMethod method, index_t num_runs,
	index_t num_folds, float64_t alpha)
{
	self->strategy->use_method(method)
		.use_num_runs(num_runs)
		.use_num_folds(num_folds)
		.use_alpha(alpha);
}

CKernelSelectionStrategy const * CMMD::get_kernel_selection_strategy() const
{
	return self->strategy.get();
}

void CMMD::add_kernel(CKernel* kernel)
{
	self->strategy->add_kernel(kernel);
}

void CMMD::select_kernel()
{
	SG_DEBUG("Entering!\n");
	auto& data_mgr=get_data_mgr();
	data_mgr.set_train_mode(true);
	CMMD::set_kernel(self->strategy->select_kernel(this));
	data_mgr.set_train_mode(false);
	SG_DEBUG("Leaving!\n");
}

void CMMD::cleanup()
{
	get_kernel_mgr().restore_kernel_at(0);
}

void CMMD::set_num_null_samples(index_t null_samples)
{
	self->num_null_samples=null_samples;
}

index_t CMMD::get_num_null_samples() const
{
	return self->num_null_samples;
}

void CMMD::set_statistic_type(EStatisticType stype)
{
	self->stype=stype;
}

EStatisticType CMMD::get_statistic_type() const
{
	return self->stype;
}

void CMMD::set_null_approximation_method(ENullApproximationMethod nmethod)
{
	self->null_approximation_method=nmethod;
}

ENullApproximationMethod CMMD::get_null_approximation_method() const
{
	return self->null_approximation_method;
}

const char* CMMD::get_name() const
{
	return "MMD";
}
