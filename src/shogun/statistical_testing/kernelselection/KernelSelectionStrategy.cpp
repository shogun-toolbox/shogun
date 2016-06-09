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

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
#include <shogun/statistical_testing/kernelselection/internals/KernelSelection.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxMeasure.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxTestPower.h>
#include <shogun/statistical_testing/kernelselection/internals/MaxXValidation.h>
#include <shogun/statistical_testing/kernelselection/internals/MedianHeuristic.h>
#include <shogun/statistical_testing/kernelselection/internals/WeightedMaxMeasure.h>
#include <shogun/statistical_testing/kernelselection/internals/WeightedMaxTestPower.h>

using namespace shogun;
using namespace internal;

struct CKernelSelectionStrategy::Self
{
	Self();

	KernelManager kernel_mgr;
	std::unique_ptr<KernelSelection> policy;

	EKernelSelectionMethod method;
	bool weighted;
	index_t num_runs;
	float64_t alpha;

	void init_policy(CMMD* estimator);

	const static EKernelSelectionMethod default_method;
	const static bool default_weighted;
	const static index_t default_num_runs;
	const static float64_t default_alpha;
};

const EKernelSelectionMethod CKernelSelectionStrategy::Self::default_method=KSM_AUTO;
const bool CKernelSelectionStrategy::Self::default_weighted=false;
const index_t CKernelSelectionStrategy::Self::default_num_runs=10;
const float64_t CKernelSelectionStrategy::Self::default_alpha=0.05;

CKernelSelectionStrategy::Self::Self() : policy(nullptr), method(default_method),
	weighted(default_weighted), num_runs(default_num_runs), alpha(default_alpha)
{
}

void CKernelSelectionStrategy::Self::init_policy(CMMD* estimator)
{
	switch (method)
	{
	case KSM_MEDIAN_HEURISTIC:
	{
		REQUIRE(!weighted, "Weighted kernel selection is not possible with MEDIAN_HEURISTIC!\n");
		policy=std::unique_ptr<MedianHeuristic>(new MedianHeuristic(kernel_mgr, estimator));
	}
	break;
	case KSM_MAXIMIZE_XVALIDATION:
	{
		REQUIRE(!weighted, "Weighted kernel selection is not possible with MAXIMIZE_XVALIDATION!\n");
		policy=std::unique_ptr<MaxXValidation>(new MaxXValidation(kernel_mgr, estimator,
			num_runs, alpha));
	}
	break;
	case KSM_MAXIMIZE_MMD:
	{
		if (weighted)
			policy=std::unique_ptr<WeightedMaxMeasure>(new WeightedMaxMeasure(kernel_mgr, estimator));
		else
			policy=std::unique_ptr<MaxMeasure>(new MaxMeasure(kernel_mgr, estimator));
	}
	break;
	case KSM_MAXIMIZE_POWER:
	{
		if (weighted)
			policy=std::unique_ptr<WeightedMaxTestPower>(new WeightedMaxTestPower(kernel_mgr, estimator));
		else
			policy=std::unique_ptr<MaxTestPower>(new MaxTestPower(kernel_mgr, estimator));
	}
	break;
	default:
	{
		SG_SERROR("Unsupported kernel selection method specified! Accepted strategies are "
			"MAXIMIZE_MMD (single, weighted), "
			"MAXIMIZE_POWER (single, weighted), "
			"MAXIMIZE_XVALIDATION (single) and "
			"MEDIAN_HEURISTIC (single)!\n");
	}
	break;
	}
}

CKernelSelectionStrategy::CKernelSelectionStrategy()
{
	init();
}

CKernelSelectionStrategy::CKernelSelectionStrategy(EKernelSelectionMethod method)
{
	init();
	self->method=method;
}

CKernelSelectionStrategy::CKernelSelectionStrategy(EKernelSelectionMethod method, bool weighted)
{
	init();
	self->method=method;
	self->weighted=weighted;
}

CKernelSelectionStrategy::CKernelSelectionStrategy(EKernelSelectionMethod method, index_t num_runs, float64_t alpha)
{
	init();
	self->method=method;
	self->num_runs=num_runs;
	self->alpha=alpha;
}

//CKernelSelectionStrategy::CKernelSelectionStrategy(const CKernelSelectionStrategy& other)
//{
//	init();
//	self->method=other.self->method;
//	self->num_runs=other.self->num_runs;
//	self->alpha=other.self->alpha;
//	for (size_t i=0; i<other.self->kernel_mgr.num_kernels(); ++i)
//		self->kernel_mgr.push_back(other.self->kernel_mgr.kernel_at(i));
//}
//
//CKernelSelectionStrategy& CKernelSelectionStrategy::operator=(const CKernelSelectionStrategy& other)
//{
//	init();
//	self->method=other.self->method;
//	self->num_runs=other.self->num_runs;
//	self->alpha=other.self->alpha;
//	for (size_t i=0; i<other.self->kernel_mgr.num_kernels(); ++i)
//		self->kernel_mgr.push_back(other.self->kernel_mgr.kernel_at(i));
//	return *this;
//}

void CKernelSelectionStrategy::init()
{
	self=std::unique_ptr<Self>(new Self());
}

CKernelSelectionStrategy::~CKernelSelectionStrategy()
{
	self->kernel_mgr.clear();
}

CKernelSelectionStrategy& CKernelSelectionStrategy::use_method(EKernelSelectionMethod method)
{
	self->method=method;
	return *this;
}

CKernelSelectionStrategy& CKernelSelectionStrategy::use_num_runs(index_t num_runs)
{
	self->num_runs=num_runs;
	return *this;
}

CKernelSelectionStrategy& CKernelSelectionStrategy::use_alpha(float64_t alpha)
{
	self->alpha=alpha;
	return *this;
}

CKernelSelectionStrategy& CKernelSelectionStrategy::use_weighted(bool weighted)
{
	self->weighted=weighted;
	return *this;
}

void CKernelSelectionStrategy::add_kernel(CKernel* kernel)
{
	self->kernel_mgr.push_back(kernel);
}

CKernel* CKernelSelectionStrategy::select_kernel(CMMD* estimator)
{
	auto num_kernels=self->kernel_mgr.num_kernels();
	REQUIRE(num_kernels>0, "Number of kernels is 0. Please add kernels using add_kernel method!\n");
	SG_DEBUG("Selecting kernels from a total of %d kernels!\n", num_kernels);

	self->init_policy(estimator);
	ASSERT(self->policy!=nullptr);

	return self->policy->select_kernel();
}

// TODO call this method when test train mode is turned off
void CKernelSelectionStrategy::erase_intermediate_results()
{
	self->policy=nullptr;
	self->kernel_mgr.clear();
}

SGMatrix<float64_t> CKernelSelectionStrategy::get_measure_matrix()
{
	REQUIRE(self->policy!=nullptr, "The kernel selection policy is not initialized!\n");
	return self->policy->get_measure_matrix();
}

SGVector<float64_t> CKernelSelectionStrategy::get_measure_vector()
{
	REQUIRE(self->policy!=nullptr, "The kernel selection policy is not initialized!\n");
	return self->policy->get_measure_vector();
}

const char* CKernelSelectionStrategy::get_name() const
{
	return "KernelSelectionStrategy";
}

const KernelManager& CKernelSelectionStrategy::get_kernel_mgr() const
{
	return self->kernel_mgr;
}
