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

#include <vector>
#include <memory>
#include <type_traits>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/Features.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/BTestMMD.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/ComputationManager.h>
#include <shogun/statistical_testing/internals/MaxMeasure.h>
#include <shogun/statistical_testing/internals/MaxTestPower.h>
#include <shogun/statistical_testing/internals/mmd/BiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedIncomplete.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockPermutation.h>

using namespace shogun;
using namespace internal;

struct CMMD::Self
{
	Self(CMMD& cmmd);

	void create_statistic_job();
	void create_variance_job();
	void create_computation_jobs();

	void merge_samples(NextSamples&, std::vector<CFeatures*>&) const;
	void compute_kernel(ComputationManager&, std::vector<CFeatures*>&, CKernel*) const;
	void compute_jobs(ComputationManager&) const;

	std::pair<float64_t, float64_t> compute_statistic_variance();
	SGVector<float64_t> sample_null();

	CMMD& owner;

	bool use_gpu;
	index_t num_null_samples;

	EStatisticType statistic_type;
	EVarianceEstimationMethod variance_estimation_method;
	ENullApproximationMethod null_approximation_method;

	std::function<float32_t(SGMatrix<float32_t>)> statistic_job;
	std::function<float32_t(SGMatrix<float32_t>)> permutation_job;
	std::function<float32_t(SGMatrix<float32_t>)> variance_job;

	KernelManager kernel_selection_mgr;
};

CMMD::Self::Self(CMMD& cmmd) : owner(cmmd),
	use_gpu(false), num_null_samples(250),
	statistic_type(EStatisticType::UNBIASED_FULL),
	variance_estimation_method(EVarianceEstimationMethod::DIRECT),
	null_approximation_method(ENullApproximationMethod::PERMUTATION),
	statistic_job(nullptr), variance_job(nullptr)
{
}

void CMMD::Self::create_computation_jobs()
{
	create_statistic_job();
	create_variance_job();
}

void CMMD::Self::create_statistic_job()
{
	const DataManager& dm=owner.get_data_manager();
	auto Bx=dm.blocksize_at(0);
	auto By=dm.blocksize_at(1);
	switch (statistic_type)
	{
		case EStatisticType::UNBIASED_FULL:
			statistic_job=mmd::UnbiasedFull(Bx);
			break;
		case EStatisticType::UNBIASED_INCOMPLETE:
			statistic_job=mmd::UnbiasedIncomplete(Bx);
			break;
		case EStatisticType::BIASED_FULL:
			statistic_job=mmd::BiasedFull(Bx);
			break;
		default : break;
	};
	permutation_job=mmd::WithinBlockPermutation(Bx, By, statistic_type);
}

void CMMD::Self::create_variance_job()
{
	switch (variance_estimation_method)
	{
		case EVarianceEstimationMethod::DIRECT:
			variance_job=owner.get_direct_estimation_method();
			break;
		case EVarianceEstimationMethod::PERMUTATION:
			variance_job=permutation_job;
			break;
		default : break;
	};
}

void CMMD::Self::merge_samples(NextSamples& next_burst, std::vector<CFeatures*>& blocks) const
{
	blocks.resize(next_burst.num_blocks());
#pragma omp parallel for
	for (size_t i=0; i<blocks.size(); ++i)
	{
		auto block_p=next_burst[0][i].get();
		auto block_q=next_burst[1][i].get();
		auto block_p_and_q=FeaturesUtil::create_merged_copy(block_p, block_q);
		blocks[i]=block_p_and_q;
	}
	next_burst.clear();
}

void CMMD::Self::compute_kernel(ComputationManager& cm, std::vector<CFeatures*>& blocks, CKernel* kernel) const
{
	REQUIRE(kernel->get_kernel_type()!=K_CUSTOM, "Underlying kernel cannot be custom!\n");
	cm.num_data(blocks.size());
#pragma omp parallel for
	for (size_t i=0; i<blocks.size(); ++i)
	{
		try
		{
			auto kernel_clone=std::unique_ptr<CKernel>(static_cast<CKernel*>(kernel->clone()));
			kernel_clone->init(blocks[i], blocks[i]);
			cm.data(i)=kernel_clone->get_kernel_matrix<float32_t>();
			kernel_clone->remove_lhs_and_rhs();
		}
		catch (ShogunException e)
		{
			SG_SERROR("%s, Try using less number of blocks per burst!\n", e.get_exception_string());
		}
	}
	blocks.resize(0);
}

void CMMD::Self::compute_jobs(ComputationManager& cm) const
{
	if (use_gpu)
		cm.use_gpu().compute_data_parallel_jobs();
	else
		cm.use_cpu().compute_data_parallel_jobs();
}

std::pair<float64_t, float64_t> CMMD::Self::compute_statistic_variance()
{
	const KernelManager& km=owner.get_kernel_manager();
	auto kernel=km.kernel_at(0);
	REQUIRE(kernel != nullptr, "Kernel is not set!\n");

	float64_t statistic=0;
	float64_t permuted_samples_statistic=0;
	float64_t variance=0;
	index_t statistic_term_counter=1;
	index_t variance_term_counter=1;

	DataManager& dm=owner.get_data_manager();
	ComputationManager cm;

	create_computation_jobs();
	cm.enqueue_job(statistic_job);
	cm.enqueue_job(variance_job);

	std::vector<CFeatures*> blocks;

	dm.start();
	auto next_burst=dm.next();
	while (!next_burst.empty())
	{
		merge_samples(next_burst, blocks);
		compute_kernel(cm, blocks, kernel);
		compute_jobs(cm);

		auto mmds=cm.result(0);
		auto vars=cm.result(1);

		for (size_t i=0; i<mmds.size(); ++i)
		{
			auto delta=mmds[i]-statistic;
			statistic+=delta/statistic_term_counter;
			statistic_term_counter++;
		}

		if (variance_estimation_method==EVarianceEstimationMethod::DIRECT)
		{
			for (size_t i=0; i<mmds.size(); ++i)
			{
				auto delta=vars[i]-variance;
				variance+=delta/variance_term_counter;
				variance_term_counter++;
			}
		}
		else
		{
			for (size_t i=0; i<vars.size(); ++i)
			{
				auto delta=vars[i]-permuted_samples_statistic;
				permuted_samples_statistic+=delta/variance_term_counter;
				variance+=delta*(vars[i]-permuted_samples_statistic);
				variance_term_counter++;
			}
		}
		next_burst=dm.next();
	}

	dm.end();
	cm.done();

	// normalize statistic and variance
	statistic=owner.normalize_statistic(statistic);
	if (variance_estimation_method==EVarianceEstimationMethod::PERMUTATION)
		variance=owner.normalize_variance(variance);

	return std::make_pair(statistic, variance);
}

SGVector<float64_t> CMMD::Self::sample_null()
{
	const KernelManager& km=owner.get_kernel_manager();
	auto kernel=km.kernel_at(0);
	REQUIRE(kernel != nullptr, "Kernel is not set!\n");

	SGVector<float64_t> statistic(num_null_samples);
	std::vector<index_t> term_counters(num_null_samples);

	std::fill(statistic.vector, statistic.vector+statistic.vlen, 0);
	std::fill(term_counters.data(), term_counters.data()+term_counters.size(), 1);

	DataManager& dm=owner.get_data_manager();
	ComputationManager cm;

	create_statistic_job();
	cm.enqueue_job(permutation_job);

	std::vector<CFeatures*> blocks;

	dm.start();
	auto next_burst=dm.next();

	while (!next_burst.empty())
	{
		merge_samples(next_burst, blocks);
		compute_kernel(cm, blocks, kernel);

		for (auto j=0; j<num_null_samples; ++j)
		{
			compute_jobs(cm);
			auto mmds=cm.result(0);
			for (size_t i=0; i<mmds.size(); ++i)
			{
				auto delta=mmds[i]-statistic[j];
				statistic[j]+=delta/term_counters[j];
				term_counters[j]++;
			}
		}
		next_burst=dm.next();
	}

	dm.end();
	cm.done();

	// normalize statistic
	std::for_each(statistic.vector, statistic.vector + statistic.vlen, [this](float64_t& value)
	{
		value=owner.normalize_statistic(value);
	});

	return statistic;
}

CMMD::CMMD() : CTwoSampleTest()
{
	self=std::unique_ptr<Self>(new Self(*this));
}

CMMD::~CMMD()
{
}

void CMMD::add_kernel(CKernel* kernel)
{
	self->kernel_selection_mgr.push_back(kernel);
}

void CMMD::select_kernel(EKernelSelectionMethod kmethod)
{
	SG_DEBUG("Entering!\n");
	SG_DEBUG("Selecting kernels from a total of %d kernels!\n", self->kernel_selection_mgr.num_kernels());
	switch (kmethod)
	{
		case EKernelSelectionMethod::MAXIMIZE_MMD:
		{
			MaxMeasure policy(self->kernel_selection_mgr, this);
			get_kernel_manager().kernel_at(0)=policy.select_kernel();
			get_kernel_manager().restore_kernel_at(0);
			break;
		}
		case EKernelSelectionMethod::MAXIMIZE_POWER:
		{
			MaxTestPower policy(self->kernel_selection_mgr, this);
			get_kernel_manager().kernel_at(0)=policy.select_kernel();
			get_kernel_manager().restore_kernel_at(0);
			break;
		}
		default:
			SG_ERROR("Unsupported kernel selection method specified!\n");
			break;
	}
	SG_DEBUG("Leaving!\n");
}

float64_t CMMD::compute_statistic()
{
	return self->compute_statistic_variance().first;
}

float64_t CMMD::compute_variance()
{
	return self->compute_statistic_variance().second;
}

std::pair<float64_t, float64_t> CMMD::compute_statistic_variance()
{
	return self->compute_statistic_variance();
}

SGVector<float64_t> CMMD::sample_null()
{
	return self->sample_null();
}

void CMMD::set_num_null_samples(index_t null_samples)
{
	self->num_null_samples=null_samples;
}

const index_t CMMD::get_num_null_samples() const
{
	return self->num_null_samples;
}

void CMMD::use_gpu(bool gpu)
{
	self->use_gpu=gpu;
}

bool CMMD::use_gpu() const
{
	return self->use_gpu;
}

void CMMD::cleanup()
{
	for (size_t i=0; i<get_kernel_manager().num_kernels(); ++i)
		get_kernel_manager().restore_kernel_at(i);
}

void CMMD::set_statistic_type(EStatisticType stype)
{
	self->statistic_type=stype;
}

const EStatisticType CMMD::get_statistic_type() const
{
	return self->statistic_type;
}

void CMMD::set_variance_estimation_method(EVarianceEstimationMethod vmethod)
{
	// TODO overload this
/*	if (std::is_same<Derived, CQuadraticTimeMMD>::value && vmethod == EVarianceEstimationMethod::PERMUTATION)
	{
		std::cerr << "cannot use permutation method for quadratic time MMD" << std::endl;
	}*/
	self->variance_estimation_method=vmethod;
}

const EVarianceEstimationMethod CMMD::get_variance_estimation_method() const
{
	return self->variance_estimation_method;
}

void CMMD::set_null_approximation_method(ENullApproximationMethod nmethod)
{
	// TODO overload this
/*	if (std::is_same<Derived, CQuadraticTimeMMD>::value && nmethod == ENullApproximationMethod::MMD1_GAUSSIAN)
	{
		std::cerr << "cannot use gaussian method for quadratic time MMD" << std::endl;
	}
	else if ((std::is_same<Derived, CBTestMMD>::value || std::is_same<Derived, CLinearTimeMMD>::value) &&
			(nmethod == ENullApproximationMethod::MMD2_SPECTRUM || nmethod == ENullApproximationMethod::MMD2_GAMMA))
	{
		std::cerr << "cannot use spectrum/gamma method for B-test/linear time MMD" << std::endl;
	}*/
	self->null_approximation_method=nmethod;
}

const ENullApproximationMethod CMMD::get_null_approximation_method() const
{
	return self->null_approximation_method;
}

const char* CMMD::get_name() const
{
	return "MMD";
}
