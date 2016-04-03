/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <utility>
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
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/ComputationManager.h>
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

	void create_computation_jobs(index_t Bx);
	void create_statistic_job(index_t Bx);
	void create_variance_job();

	void merge_samples(NextSamples&, std::vector<std::shared_ptr<CFeatures>>&) const;
	void compute_kernel(ComputationManager&, std::vector<std::shared_ptr<CFeatures>>&, CKernel*) const;
	void compute_jobs(ComputationManager&) const;

	std::pair<float64_t, float64_t> compute_statistic_variance();
	SGVector<float64_t> sample_null();

	CMMD& owner;

	bool use_gpu_for_computation;
	index_t num_null_samples;

	EStatisticType statistic_type;
	EVarianceEstimationMethod variance_estimation_method;
	ENullApproximationMethod null_approximation_method;

	std::function<float64_t(SGMatrix<float64_t>)> statistic_job;
	std::function<float64_t(SGMatrix<float64_t>)> permutation_job;
	std::function<float64_t(SGMatrix<float64_t>)> variance_job;
};

CMMD::Self::Self(CMMD& cmmd) : owner(cmmd),
	use_gpu_for_computation(false), num_null_samples(250),
	statistic_type(EStatisticType::UNBIASED_FULL),
	variance_estimation_method(EVarianceEstimationMethod::DIRECT),
	null_approximation_method(ENullApproximationMethod::PERMUTATION),
	statistic_job(nullptr), variance_job(nullptr)
{
}

void CMMD::Self::create_computation_jobs(index_t Bx)
{
	create_statistic_job(Bx);
	create_variance_job();
}

void CMMD::Self::create_statistic_job(index_t Bx)
{
	switch (statistic_type)
	{
		case EStatisticType::UNBIASED_FULL:
			statistic_job = mmd::UnbiasedFull(Bx);
			permutation_job = mmd::WithinBlockPermutation<mmd::UnbiasedFull>(Bx);
			break;
		case EStatisticType::UNBIASED_INCOMPLETE:
			statistic_job = mmd::UnbiasedIncomplete(Bx);
			permutation_job = mmd::WithinBlockPermutation<mmd::UnbiasedIncomplete>(Bx);
			break;
		case EStatisticType::BIASED_FULL:
			statistic_job = mmd::BiasedFull(Bx);
			permutation_job = mmd::WithinBlockPermutation<mmd::BiasedFull>(Bx);
			break;
		default : break;
	};
}

void CMMD::Self::create_variance_job()
{
	switch (variance_estimation_method)
	{
		case EVarianceEstimationMethod::DIRECT:
			variance_job = owner.get_direct_estimation_method();
			break;
		case EVarianceEstimationMethod::PERMUTATION:
			variance_job = permutation_job;
			break;
		default : break;
	};
}

#define get_block_p(i) next_burst[0][i]
#define get_block_q(i) next_burst[1][i]
void CMMD::Self::merge_samples(NextSamples& next_burst, std::vector<std::shared_ptr<CFeatures>>& blocks) const
{
	blocks.resize(next_burst.num_blocks());

#pragma omp parallel for
	for (size_t i = 0; i < blocks.size(); ++i)
	{
		auto block_p = get_block_p(i);
		auto block_q = get_block_q(i);

		auto block_p_and_q = block_p->create_merged_copy(block_q.get());
		SG_REF(block_p_and_q);

		block_p = nullptr;
		block_q = nullptr;

		blocks[i] = std::shared_ptr<CFeatures>(block_p_and_q, [](CFeatures* ptr) { SG_UNREF(ptr); });
	}
}
#undef get_block_p
#undef get_block_q

void CMMD::Self::compute_kernel(ComputationManager& cm, std::vector<std::shared_ptr<CFeatures>>& blocks, CKernel* kernel) const
{
	cm.num_data(blocks.size());

#pragma omp parallel for
	for (size_t i = 0; i < blocks.size(); ++i)
	{
		try
		{
			auto kernel_clone = std::unique_ptr<CKernel>(static_cast<CKernel*>(kernel->clone()));
			kernel_clone->init(blocks[i].get(), blocks[i].get());
			cm.data(i) = std::unique_ptr<CCustomKernel>(new CCustomKernel(kernel_clone.get()))->get_kernel_matrix();
			kernel_clone->remove_lhs_and_rhs();
		}
		catch (ShogunException e)
		{
			SG_SERROR("%s, Try using less number of blocks per burst!\n", e.get_exception_string());
		}
	}
}

void CMMD::Self::compute_jobs(ComputationManager& cm) const
{
	if (use_gpu_for_computation)
	{
		cm.use_gpu().compute();
	}
	else
	{
		cm.use_cpu().compute();
	}
}

std::pair<float64_t, float64_t> CMMD::Self::compute_statistic_variance()
{
	DataManager& dm = owner.get_data_manager();
	const KernelManager& km = owner.get_kernel_manager();

	float64_t statistic = 0;
	float64_t permuted_samples_statistic = 0;
	float64_t variance = 0;

	auto kernel = km.kernel_at(0);
	REQUIRE(kernel != nullptr, "Kernel is not set!\n");

	index_t term_counters = 1;

	dm.start();
	auto next_burst = dm.next();

	create_computation_jobs(owner.get_data_manager().blocksize_at(0));

	ComputationManager cm;
	// enqueue statistic and variance computation jobs on the computed kernel matrices
	cm.enqueue_job(statistic_job);
	cm.enqueue_job(variance_job);

	std::vector<std::shared_ptr<CFeatures>> blocks;

	while (!next_burst.empty())
	{
		merge_samples(next_burst, blocks);
		compute_kernel(cm, blocks, kernel);
		compute_jobs(cm);

		auto mmds = cm.next_result();
		auto vars = cm.next_result();

		for (size_t i = 0; i < mmds.size(); ++i)
		{
			auto delta = mmds[i] - statistic;
			statistic += delta / term_counters;
		}

		if (variance_estimation_method == EVarianceEstimationMethod::DIRECT)
		{
			for (size_t i = 0; i < mmds.size(); ++i)
			{
				auto delta = vars[i] - variance;
				variance += delta / term_counters;
			}
		}
		else
		{
			for (size_t i = 0; i < mmds.size(); ++i)
			{
				auto delta = vars[i] - permuted_samples_statistic;
				permuted_samples_statistic += delta / term_counters;
				variance += delta * (vars[i] - permuted_samples_statistic);
			}
		}
		term_counters++;

		next_burst = dm.next();
	}

	dm.end();
	cm.done();

	// normalize statistic and variance
	statistic = owner.normalize_statistic(statistic);

	if (variance_estimation_method == EVarianceEstimationMethod::PERMUTATION)
	{
		variance = owner.normalize_variance(variance);
	}

	return std::make_pair(statistic, variance);
}

SGVector<float64_t> CMMD::Self::sample_null()
{
	DataManager& dm = owner.get_data_manager();
	const KernelManager& km = owner.get_kernel_manager();

	SGVector<float64_t> statistic(num_null_samples);
	std::fill(statistic.vector, statistic.vector + statistic.vlen, 0);

	auto kernel = km.kernel_at(0);
	REQUIRE(kernel != nullptr, "Kernel is not set!\n");

	std::vector<index_t> term_counters(num_null_samples);
	std::fill(term_counters.data(), term_counters.data() + term_counters.size(), 1);

	dm.start();
	auto next_burst = dm.next();

	create_statistic_job(owner.get_data_manager().blocksize_at(0));

	ComputationManager cm;
	cm.enqueue_job(permutation_job);

	std::vector<std::shared_ptr<CFeatures>> blocks;

	while (!next_burst.empty())
	{
		merge_samples(next_burst, blocks);
		compute_kernel(cm, blocks, kernel);

		for (auto j = 0; j < num_null_samples; ++j)
		{
			compute_jobs(cm);
			auto mmds = cm.next_result();
			for (size_t i = 0; i < mmds.size(); ++i)
			{
				auto delta = mmds[i] - statistic[j];
				statistic[j] += delta / term_counters[j];
			}

			term_counters[j]++;
		}

		next_burst = dm.next();
	}

	dm.end();
	cm.done();

	// normalize statistic
	std::for_each(statistic.vector, statistic.vector + statistic.vlen, [this](float64_t& value)
	{
		value = owner.normalize_statistic(value);
	});

	return statistic;
}

CMMD::CMMD() : CTwoSampleTest()
{
	self = std::unique_ptr<Self>(new Self(*this));
}

CMMD::~CMMD()
{
}

float64_t CMMD::compute_statistic()
{
	return self->compute_statistic_variance().first;
}

float64_t CMMD::compute_variance()
{
	return self->compute_statistic_variance().second;
}

SGVector<float64_t> CMMD::sample_null()
{
	return self->sample_null();
}

void CMMD::set_num_null_samples(index_t null_samples)
{
	self->num_null_samples = null_samples;
}

const index_t CMMD::get_num_null_samples() const
{
	return self->num_null_samples;
}

void CMMD::use_gpu(bool gpu)
{
	self->use_gpu_for_computation = gpu;
}

void CMMD::set_statistic_type(EStatisticType stype)
{
	self->statistic_type = stype;
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
	self->variance_estimation_method = vmethod;
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
	self->null_approximation_method = nmethod;
}

const ENullApproximationMethod CMMD::get_null_approximation_method() const
{
	return self->null_approximation_method;
}

const char* CMMD::get_name() const
{
	return "MMD";
}
