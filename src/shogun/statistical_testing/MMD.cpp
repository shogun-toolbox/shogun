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
	void create_variance_job(index_t Bx);

	std::pair<SGVector<float64_t>, SGVector<float64_t>> compute_statistic_variance();

	CMMD& owner;

	bool use_gpu_for_computation;
	bool simulate_null;
	index_t num_null_samples;
	EStatisticType statistic_type;
	EVarianceEstimationMethod variance_estimation_method;
	ENullApproximationMethod null_approximation_method;

	std::function<float64_t(SGMatrix<float64_t>)> statistic_job;
	std::function<float64_t(SGMatrix<float64_t>)> variance_job;
};

CMMD::Self::Self(CMMD& cmmd) : owner(cmmd),
	use_gpu_for_computation(false), simulate_null(false), num_null_samples(0),
	statistic_type(EStatisticType::UNBIASED_FULL), variance_estimation_method(EVarianceEstimationMethod::DIRECT),
	null_approximation_method(ENullApproximationMethod::PERMUTATION), statistic_job(nullptr), variance_job(nullptr)
{
}

void CMMD::Self::create_computation_jobs(index_t Bx)
{
	create_statistic_job(Bx);
	create_variance_job(Bx);
}

void CMMD::Self::create_statistic_job(index_t Bx)
{
	switch (statistic_type)
	{
		case EStatisticType::UNBIASED_FULL:
			statistic_job = mmd::UnbiasedFull(Bx);
			break;
		case EStatisticType::UNBIASED_INCOMPLETE:
			statistic_job = mmd::UnbiasedIncomplete(Bx);
			break;
		case EStatisticType::BIASED_FULL:
			statistic_job = mmd::BiasedFull(Bx);
			break;
		default : break;
	};
}

void CMMD::Self::create_variance_job(index_t Bx)
{
	switch (variance_estimation_method)
	{
		case EVarianceEstimationMethod::DIRECT:
			variance_job = owner.get_direct_estimation_method();
			break;
		case EVarianceEstimationMethod::PERMUTATION:
			switch(statistic_type)
			{
				case EStatisticType::UNBIASED_FULL:
					variance_job = mmd::WithinBlockPermutation<mmd::UnbiasedFull>(Bx);
					break;
				case EStatisticType::UNBIASED_INCOMPLETE:
					variance_job = mmd::WithinBlockPermutation<mmd::UnbiasedIncomplete>(Bx);
					break;
				case EStatisticType::BIASED_FULL:
					variance_job = mmd::WithinBlockPermutation<mmd::BiasedFull>(Bx);
					break;
				default : break;
			}
			break;
		default : break;
	};
}

std::pair<SGVector<float64_t>, SGVector<float64_t>> CMMD::Self::compute_statistic_variance()
{
	DataManager& dm = owner.get_data_manager();
	const KernelManager& km = owner.get_kernel_manager();

	SGVector<float64_t> statistic;
	SGVector<float64_t> stat_perm;
	SGVector<float64_t> variance;

	auto kernel = km.kernel_at(0);
	ASSERT(kernel != nullptr);
	auto num_kernels = 1;

	std::vector<CKernel*> kernels;

	if (kernel->get_kernel_type() == K_COMBINED)
	{
		auto combined_kernel = static_cast<CCombinedKernel*>(kernel);
		num_kernels = combined_kernel->get_num_subkernels();

		kernels = std::vector<CKernel*>(num_kernels);
		for (auto i = 0; i < num_kernels; ++i)
		{
			kernels[i] = combined_kernel->get_kernel(i);
		}
	}
	else
	{
		kernels.push_back(kernel);
	}

	statistic = SGVector<float64_t>(num_kernels);
	stat_perm = SGVector<float64_t>(num_kernels);
	variance = SGVector<float64_t>(num_kernels);

	std::fill(statistic.vector, statistic.vector + statistic.vlen, 0);
	std::fill(stat_perm.vector, stat_perm.vector + stat_perm.vlen, 0);
	std::fill(variance.vector, variance.vector + variance.vlen, 0);

	std::vector<index_t> term_counters(statistic.vlen);
	std::fill(term_counters.data(), term_counters.data() + term_counters.size(), 1);

	ComputationManager cm;
	dm.start();
	auto next_burst = dm.next();

	create_computation_jobs(owner.get_data_manager().blocksize_at(0));

	std::vector<std::shared_ptr<CFeatures>> blocks;

	while (!next_burst.empty())
	{
		cm.num_data(next_burst.num_blocks());
		blocks.resize(next_burst.num_blocks());

#pragma omp parallel for
		for (auto i = 0; i < next_burst.num_blocks(); ++i)
		{
			auto block_p = next_burst[0][i];
			auto block_q = next_burst[1][i];

			auto block_p_q = block_p->create_merged_copy(block_q.get());
			SG_REF(block_p_q);
			if (simulate_null)
			{
				SGVector<index_t> inds(block_p_q->get_num_vectors());
				std::iota(inds.vector, inds.vector + inds.vlen, 0);
				CMath::permute(inds);
				block_p_q->add_subset(inds);
			}

			block_p = nullptr;
			block_q = nullptr;

			blocks[i] = std::shared_ptr<CFeatures>(block_p_q, [](CFeatures* ptr) { SG_UNREF(ptr); });
		}

		for (auto i = 0; i < kernels.size(); ++i)
		{
#pragma omp parallel for
			for (auto j = 0; j < blocks.size(); ++j)
			{
				try
				{
					auto curr_kernel = std::unique_ptr<CKernel>(static_cast<CKernel*>(kernels[i]->clone()));
					curr_kernel->init(blocks[j].get(), blocks[j].get());
					cm.data(j) = std::unique_ptr<CCustomKernel>(new CCustomKernel(curr_kernel.get()))->get_kernel_matrix();
					curr_kernel->remove_lhs_and_rhs();
				}
				catch (ShogunException e)
				{
					SG_SERROR("%s, Try using less number of blocks per burst!\n", e.get_exception_string());
				}
			}

			// enqueue statistic and variance computation jobs on the computed kernel matrices
			cm.enqueue_job(statistic_job);
			cm.enqueue_job(variance_job);

			if (use_gpu_for_computation)
			{
				cm.use_gpu().compute();
			}
			else
			{
				cm.use_cpu().compute();
			}

			auto mmds = cm.next_result();
			auto vars = cm.next_result();

			for (auto j = 0; j < mmds.size(); ++j)
			{
				auto delta = mmds[j] - statistic[i];
				statistic[i] += delta / term_counters[i];
			}

			if (variance_estimation_method == EVarianceEstimationMethod::DIRECT)
			{
				for (auto j = 0; j < mmds.size(); ++j)
				{
					auto delta = vars[j] - variance[i];
					variance[i] += delta / term_counters[i];
				}
			}
			else
			{
				for (auto j = 0; j < mmds.size(); ++j)
				{
					auto delta = vars[j] - stat_perm[i];
					stat_perm[i] += delta / term_counters[i];
					variance[i] += delta * (vars[j] - stat_perm[i]);
				}
			}
			term_counters[i]++;
		}

		next_burst = dm.next();
	}

	dm.end();

	// normalize statistic and variance
	std::for_each(statistic.vector, statistic.vector + statistic.vlen, [this](float64_t& v)
	{
		v = owner.normalize_statistic(v);
	});

	if (variance_estimation_method == EVarianceEstimationMethod::PERMUTATION)
	{
		std::for_each(variance.vector, variance.vector + variance.vlen, [this](float64_t& v)
		{
			v = owner.normalize_variance(v);
		});
	}

	return std::make_pair(statistic, variance);
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
	return self->compute_statistic_variance().first[0];
}

float64_t CMMD::compute_variance()
{
	return self->compute_statistic_variance().second[0];
}

SGVector<float64_t> CMMD::compute_statistic(bool multiple_kernels)
{
	if (multiple_kernels)
	{
		const KernelManager& km = get_kernel_manager();
		auto kernel = km.kernel_at(0);
		ASSERT(kernel->get_kernel_type() == K_COMBINED);
	}
	return self->compute_statistic_variance().first;
}

SGVector<float64_t> CMMD::compute_variance(bool multiple_kernels)
{
	if (multiple_kernels)
	{
		const KernelManager& km = get_kernel_manager();
		auto kernel = km.kernel_at(0);
		ASSERT(kernel->get_kernel_type() == K_COMBINED);
	}
	return self->compute_statistic_variance().second;
}

SGVector<float64_t> CMMD::sample_null()
{
	SGVector<float64_t> null_samples(self->num_null_samples);
	auto old = self->simulate_null;
	self->simulate_null = true;
	for (auto i = 0; i < self->num_null_samples; ++i)
	{
		null_samples[i] = compute_statistic();
	}
	self->simulate_null = old;
	return null_samples;
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

void CMMD::set_simulate_null(bool simulate_null)
{
	self->simulate_null = simulate_null;
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
