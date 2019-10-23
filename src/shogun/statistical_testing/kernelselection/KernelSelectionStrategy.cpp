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

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/StreamingMMD.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
#include <shogun/statistical_testing/kernelselection/internals/KernelSelection.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxMeasure.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxTestPower.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxCrossValidation.h>
#include <shogun/statistical_testing/kernelselection/internals/MedianHeuristic.h>

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/statistical_testing/kernelselection/internals/WeightedMaxMeasure.h>
#include <shogun/statistical_testing/kernelselection/internals/WeightedMaxTestPower.h>

#include <utility>
#endif //USE_GPL_SHOGUN

using namespace shogun;
using namespace internal;

struct KernelSelectionStrategy::Self
{
	Self(KernelSelectionStrategy::prng_type& _prng);

	KernelManager kernel_mgr;
	std::unique_ptr<KernelSelection> policy;

	EKernelSelectionMethod method;
	bool weighted;
	index_t num_runs;
	index_t num_folds;
	float64_t alpha;

	void init_policy(std::shared_ptr<MMD> estimator);

	const static EKernelSelectionMethod default_method;
	const static bool default_weighted;
	const static index_t default_num_runs;
	const static index_t default_num_folds;
	const static float64_t default_alpha;

	KernelSelectionStrategy::prng_type& prng;
};

const EKernelSelectionMethod KernelSelectionStrategy::Self::default_method=KSM_AUTO;
const bool KernelSelectionStrategy::Self::default_weighted=false;
const index_t KernelSelectionStrategy::Self::default_num_runs=10;
const index_t KernelSelectionStrategy::Self::default_num_folds=3;
const float64_t KernelSelectionStrategy::Self::default_alpha=0.05;

KernelSelectionStrategy::Self::Self(KernelSelectionStrategy::prng_type& _prng) : policy(nullptr), method(default_method),
	weighted(default_weighted), num_runs(default_num_runs), num_folds(default_num_folds), alpha(default_alpha), prng(_prng)
{
}

void KernelSelectionStrategy::Self::init_policy(std::shared_ptr<MMD> estimator)
{
	switch (method)
	{
	case KSM_MEDIAN_HEURISTIC:
	{
		require(!weighted, "Weighted kernel selection is not possible with MEDIAN_HEURISTIC!");
		policy=std::make_unique<MedianHeuristic>(kernel_mgr, estimator);
	}
	break;
	case KSM_CROSS_VALIDATION:
	{
		require(!weighted, "Weighted kernel selection is not possible with CROSS_VALIDATION!");
		using prng_type = KernelSelectionStrategy::prng_type;
		policy=std::make_unique<MaxCrossValidation<prng_type>>(kernel_mgr, estimator,
			num_runs, num_folds, alpha, prng);
	}
	break;
	case KSM_MAXIMIZE_MMD:
	{
		if (weighted)
			#ifdef USE_GPL_SHOGUN
			policy=std::make_unique<WeightedMaxMeasure>(kernel_mgr, estimator);
			#else
			gpl_only(SOURCE_LOCATION);
			#endif // USE_GPL_SHOGUN
		else
			policy=std::make_unique<MaxMeasure>(kernel_mgr, estimator);
	}
	break;
	case KSM_MAXIMIZE_POWER:
	{
		if (weighted)
		{
			#ifdef USE_GPL_SHOGUN
			auto casted_estimator=estimator->as<StreamingMMD>();
			require(casted_estimator, "Weighted kernel selection is not possible with MAXIMIZE_POWER!");
			policy=std::make_unique<WeightedMaxTestPower>(kernel_mgr, estimator);
			#else
			gpl_only(SOURCE_LOCATION);
			#endif // USE_GPL_SHOGUN
		}
		else
			policy=std::make_unique<MaxTestPower>(kernel_mgr, estimator);
	}
	break;
	default:
	{
		error("Unsupported kernel selection method specified! Accepted strategies are "
			"MAXIMIZE_MMD (single, weighted), "
			"MAXIMIZE_POWER (single, weighted), "
			"CROSS_VALIDATION (single) and "
			"MEDIAN_HEURISTIC (single)!");
	}
	break;
	}
}

KernelSelectionStrategy::KernelSelectionStrategy()
{
	init();
}

KernelSelectionStrategy::KernelSelectionStrategy(EKernelSelectionMethod method, bool weighted)
{
	init();
	self->method=method;
	self->weighted=weighted;
}

KernelSelectionStrategy::KernelSelectionStrategy(EKernelSelectionMethod method, index_t num_runs,
	index_t num_folds, float64_t alpha)
{
	init();
	self->method=method;
	self->num_runs=num_runs;
	self->num_folds=num_folds;
	self->alpha=alpha;
}

void KernelSelectionStrategy::init()
{
	self=std::make_unique<Self>(m_prng);
}

KernelSelectionStrategy::~KernelSelectionStrategy()
{
	self->kernel_mgr.clear();
}

KernelSelectionStrategy& KernelSelectionStrategy::use_method(EKernelSelectionMethod method)
{
	self->method=method;
	return *this;
}

KernelSelectionStrategy& KernelSelectionStrategy::use_num_runs(index_t num_runs)
{
	self->num_runs=num_runs;
	return *this;
}

KernelSelectionStrategy& KernelSelectionStrategy::use_num_folds(index_t num_folds)
{
	self->num_folds=num_folds;
	return *this;
}

KernelSelectionStrategy& KernelSelectionStrategy::use_alpha(float64_t alpha)
{
	self->alpha=alpha;
	return *this;
}

KernelSelectionStrategy& KernelSelectionStrategy::use_weighted(bool weighted)
{
	self->weighted=weighted;
	return *this;
}

EKernelSelectionMethod KernelSelectionStrategy::get_method() const
{
	return self->method;
}

index_t KernelSelectionStrategy::get_num_runs() const
{
	return self->num_runs;
}

index_t KernelSelectionStrategy::get_num_folds() const
{
	return self->num_folds;
}

float64_t KernelSelectionStrategy::get_alpha() const
{
	return self->alpha;
}

bool KernelSelectionStrategy::get_weighted() const
{
	return self->weighted;
}

void KernelSelectionStrategy::add_kernel(std::shared_ptr<Kernel> kernel)
{
	self->kernel_mgr.push_back(std::move(kernel));
}

std::shared_ptr<Kernel> KernelSelectionStrategy::select_kernel(std::shared_ptr<MMD> estimator)
{
	auto num_kernels=self->kernel_mgr.num_kernels();
	require(num_kernels>0, "Number of kernels is 0. Please add kernels using add_kernel method!");
	SG_DEBUG("Selecting kernels from a total of {} kernels!", num_kernels);

	self->init_policy(std::move(estimator));
	ASSERT(self->policy!=nullptr);

	return self->policy->select_kernel();
}

// TODO call this method when test train mode is turned off
void KernelSelectionStrategy::erase_intermediate_results()
{
	self->policy=nullptr;
	self->kernel_mgr.clear();
}

SGMatrix<float64_t> KernelSelectionStrategy::get_measure_matrix()
{
	require(self->policy!=nullptr, "The kernel selection policy is not initialized!");
	return self->policy->get_measure_matrix();
}

SGVector<float64_t> KernelSelectionStrategy::get_measure_vector()
{
	require(self->policy!=nullptr, "The kernel selection policy is not initialized!");
	return self->policy->get_measure_vector();
}

const char* KernelSelectionStrategy::get_name() const
{
	return "KernelSelectionStrategy";
}

const KernelManager& KernelSelectionStrategy::get_kernel_mgr() const
{
	return self->kernel_mgr;
}
