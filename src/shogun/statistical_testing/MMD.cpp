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

namespace shogun
{
EStatisticType statistic_type(machine_int_t method)
{
	if (method == ST_UNBIASED_FULL)
		return ST_UNBIASED_FULL;
	else if (method == ST_UNBIASED_INCOMPLETE)
		return ST_UNBIASED_INCOMPLETE;
	else if (method == ST_BIASED_FULL)
		return ST_BIASED_FULL;
	else
	{
		SG_SERROR("Unknown statistic type %d.\n", method);
		return ST_UNBIASED_FULL;
	}
}

EVarianceEstimationMethod variance_estimation_method(machine_int_t method)
{
	if (method == VEM_DIRECT)
		return VEM_DIRECT;
	else if (method == VEM_PERMUTATION)
		return VEM_PERMUTATION;
	else
	{
		SG_SERROR("Unknown variance estimation method %d.\n", method);
		return VEM_PERMUTATION;
	}
}

ENullApproximationMethod null_approximation_method(machine_int_t method)
{
	if (method == NAM_PERMUTATION)
		return NAM_PERMUTATION;
	else if (method == NAM_MMD1_GAUSSIAN)
		return NAM_MMD1_GAUSSIAN;
	else if (method == NAM_MMD2_SPECTRUM)
		return NAM_MMD2_SPECTRUM;
	else if (method == NAM_MMD2_GAMMA)
		return NAM_MMD2_GAMMA;
	else
	{
		SG_SERROR("Unknown null approximation method %d.\n", method);
		return NAM_MMD2_GAMMA;
	}
}


EKernelSelectionMethod kernel_selection_method(machine_int_t method)
{
	if (method == KSM_MEDIAN_HEURISTIC)
		return KSM_MEDIAN_HEURISTIC;
	else if (method == KSM_MAXIMIZE_MMD)
		return KSM_MAXIMIZE_MMD;
	else if (method == KSM_MAXIMIZE_POWER)
		return KSM_MAXIMIZE_POWER;
	else if (method == KSM_CROSS_VALIDATION)
		return KSM_CROSS_VALIDATION;
	else
	{
		SG_SERROR("Unknown kernel selection method %d.\n", method);
		return KSM_AUTO;
	}
}
}

struct MMD::Self
{
	Self()
	{
		num_null_samples = DEFAULT_NUM_NULL_SAMPLES;
		stype = DEFAULT_STYPE;
		null_approximation_method = DEFAULT_NULL_APPROXIMATION_METHOD;
		strategy=std::make_shared<KernelSelectionStrategy>();
	}

	~Self()
	{
	}

	index_t num_null_samples;
	EStatisticType stype;
	ENullApproximationMethod null_approximation_method;
	std::shared_ptr<KernelSelectionStrategy> strategy;

	static constexpr index_t DEFAULT_NUM_NULL_SAMPLES = 250;
	static constexpr EStatisticType DEFAULT_STYPE = ST_UNBIASED_FULL;
	static constexpr ENullApproximationMethod DEFAULT_NULL_APPROXIMATION_METHOD = NAM_PERMUTATION;
};

MMD::MMD() : RandomMixin<TwoSampleTest>()
{
	init();
}

void MMD::init()
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	self=std::make_unique<Self>();
	watch_param("strategy", &(self->strategy));
}

MMD::~MMD()
{
	cleanup();
}

void MMD::set_kernel_selection_strategy(machine_int_t method, bool weighted)
{
	self->strategy->use_method(kernel_selection_method(method))
		.use_weighted(weighted);
}

void MMD::set_kernel_selection_strategy(machine_int_t method, index_t num_runs,
	index_t num_folds, float64_t alpha)
{
	self->strategy->use_method(kernel_selection_method(method))
		.use_num_runs(num_runs)
		.use_num_folds(num_folds)
		.use_alpha(alpha);
}

KernelSelectionStrategy const* MMD::get_kernel_selection_strategy() const
{
	return self->strategy.get();
}

void MMD::add_kernel(std::shared_ptr<Kernel> kernel)
{
	self->strategy->add_kernel(kernel);
}

void MMD::select_kernel()
{
	SG_DEBUG("Entering!\n");
	auto& data_mgr=get_data_mgr();
	data_mgr.set_train_mode(true);
	MMD::set_kernel(self->strategy->select_kernel(shared_from_this()->as<MMD>()));
	data_mgr.set_train_mode(false);
	SG_DEBUG("Leaving!\n");
}

void MMD::cleanup()
{
	get_kernel_mgr().restore_kernel_at(0);
}

void MMD::set_num_null_samples(index_t null_samples)
{
	self->num_null_samples=null_samples;
}

index_t MMD::get_num_null_samples() const
{
	return self->num_null_samples;
}

void MMD::set_statistic_type(machine_int_t stype)
{
	self->stype=statistic_type(stype);
}

EStatisticType MMD::get_statistic_type() const
{
	return self->stype;
}

void MMD::set_null_approximation_method(machine_int_t nmethod)
{
	self->null_approximation_method=null_approximation_method(nmethod);
}

ENullApproximationMethod MMD::get_null_approximation_method() const
{
	return self->null_approximation_method;
}

const char* MMD::get_name() const
{
	return "MMD";
}
