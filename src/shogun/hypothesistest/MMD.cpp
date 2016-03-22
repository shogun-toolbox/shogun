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
#include <iostream>
#include <type_traits>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/Features.h>
#include <shogun/hypothesistest/MMD.h>
#include <shogun/hypothesistest/QuadraticTimeMMD.h>
#include <shogun/hypothesistest/BTestMMD.h>
#include <shogun/hypothesistest/LinearTimeMMD.h>
#include <shogun/hypothesistest/internals/NextSamples.h>
#include <shogun/hypothesistest/internals/DataManager.h>
#include <shogun/hypothesistest/internals/KernelManager.h>
#include <shogun/hypothesistest/internals/ComputationManager.h>
#include <shogun/hypothesistest/internals/mmd/BiasedFull.h>
#include <shogun/hypothesistest/internals/mmd/UnbiasedFull.h>
#include <shogun/hypothesistest/internals/mmd/UnbiasedIncomplete.h>
#include <shogun/hypothesistest/internals/mmd/WithinBlockDirect.h>
#include <shogun/hypothesistest/internals/mmd/WithinBlockPermutation.h>

using namespace shogun;
using namespace internal;

template <class Derived>
struct CMMD<Derived>::Self
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
	S_TYPE statistic_type;
	V_METHOD variance_estimation_method;
	N_METHOD null_approximation_method;

	std::function<float64_t(SGMatrix<float64_t>)> statistic_job;
	std::function<float64_t(SGMatrix<float64_t>)> variance_job;
};

template <class Derived>
CMMD<Derived>::Self::Self(CMMD& cmmd) : owner(cmmd),
	use_gpu_for_computation(false), simulate_null(false), num_null_samples(0),
	statistic_type(S_TYPE::UNBIASED_FULL), variance_estimation_method(V_METHOD::DIRECT),
	null_approximation_method(N_METHOD::PERMUTATION), statistic_job(nullptr), variance_job(nullptr)
{
}

template <class Derived>
void CMMD<Derived>::Self::create_computation_jobs(index_t Bx)
{
	create_statistic_job(Bx);
	create_variance_job(Bx);
}

template <class Derived>
void CMMD<Derived>::Self::create_statistic_job(index_t Bx)
{
	switch (statistic_type)
	{
		case S_TYPE::UNBIASED_FULL:
			statistic_job = mmd::UnbiasedFull(Bx);
			break;
		case S_TYPE::UNBIASED_INCOMPLETE:
			statistic_job = mmd::UnbiasedIncomplete(Bx);
			break;
		case S_TYPE::BIASED_FULL:
			statistic_job = mmd::BiasedFull(Bx);
			break;
		default : break;
	};
}

template <class Derived>
void CMMD<Derived>::Self::create_variance_job(index_t Bx)
{
	switch (variance_estimation_method)
	{
		case V_METHOD::DIRECT:
			variance_job = Derived::get_direct_estimation_method();
			break;
		case V_METHOD::PERMUTATION:
			switch(statistic_type)
			{
				case S_TYPE::UNBIASED_FULL:
					variance_job = mmd::WithinBlockPermutation<mmd::UnbiasedFull>(Bx);
					break;
				case S_TYPE::UNBIASED_INCOMPLETE:
					variance_job = mmd::WithinBlockPermutation<mmd::UnbiasedIncomplete>(Bx);
					break;
				case S_TYPE::BIASED_FULL:
					variance_job = mmd::WithinBlockPermutation<mmd::BiasedFull>(Bx);
					break;
				default : break;
			}
			break;
		default : break;
	};
}

template <class Derived>
std::pair<SGVector<float64_t>, SGVector<float64_t>> CMMD<Derived>::Self::compute_statistic_variance()
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
					std::cerr << e.get_exception_string() << std::endl;
					std::cerr << "Try using less number of blocks per burst" << std::endl;
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

			if (variance_estimation_method == V_METHOD::DIRECT)
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
		v = static_cast<Derived*>(&owner)->normalize_statistic(v);
	});

	if (variance_estimation_method == V_METHOD::PERMUTATION)
	{
		std::for_each(variance.vector, variance.vector + variance.vlen, [this](float64_t& v)
		{
			v = static_cast<Derived*>(&owner)->normalize_variance(v);
		});
	}

	return std::make_pair(statistic, variance);
}

template <class Derived>
CMMD<Derived>::CMMD() : CTwoSampleTest()
{
	CSGObject::set_generic<CSGObject*>();
	self = std::unique_ptr<Self>(new Self(*this));
}

template <class Derived>
CMMD<Derived>::~CMMD()
{
}

template <class Derived>
float64_t CMMD<Derived>::compute_statistic()
{
	return self->compute_statistic_variance().first[0];
}

template <class Derived>
float64_t CMMD<Derived>::compute_variance()
{
	return self->compute_statistic_variance().second[0];
}

template <class Derived>
SGVector<float64_t> CMMD<Derived>::compute_statistic(bool multiple_kernels)
{
	if (multiple_kernels)
	{
		const KernelManager& km = get_kernel_manager();
		auto kernel = km.kernel_at(0);
		ASSERT(kernel->get_kernel_type() == K_COMBINED);
	}
	return self->compute_statistic_variance().first;
}

template <class Derived>
SGVector<float64_t> CMMD<Derived>::compute_variance(bool multiple_kernels)
{
	if (multiple_kernels)
	{
		const KernelManager& km = get_kernel_manager();
		auto kernel = km.kernel_at(0);
		ASSERT(kernel->get_kernel_type() == K_COMBINED);
	}
	return self->compute_statistic_variance().second;
}

template <class Derived>
SGVector<float64_t> CMMD<Derived>::sample_null()
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

template <class Derived>
void CMMD<Derived>::set_num_null_samples(index_t null_samples)
{
	self->num_null_samples = null_samples;
}

template <class Derived>
const index_t CMMD<Derived>::get_num_null_samples() const
{
	return self->num_null_samples;
}

template <class Derived>
void CMMD<Derived>::use_gpu(bool gpu)
{
	self->use_gpu_for_computation = gpu;
}

template <class Derived>
void CMMD<Derived>::set_simulate_null(bool null)
{
	self->simulate_null = null;
}

template <class Derived>
void CMMD<Derived>::set_statistic_type(S_TYPE stype)
{
	self->statistic_type = stype;
}

template <class Derived>
const S_TYPE CMMD<Derived>::get_statistic_type() const
{
	return self->statistic_type;
}

template <class Derived>
void CMMD<Derived>::set_variance_estimation_method(V_METHOD vmethod)
{
	if (std::is_same<Derived, CQuadraticTimeMMD>::value && vmethod == V_METHOD::PERMUTATION)
	{
		std::cerr << "cannot use permutation method for quadratic time MMD" << std::endl;
	}
	self->variance_estimation_method = vmethod;
}

template <class Derived>
const V_METHOD CMMD<Derived>::get_variance_estimation_method() const
{
	return self->variance_estimation_method;
}

template <class Derived>
void CMMD<Derived>::set_null_approximation_method(N_METHOD nmethod)
{
	if (std::is_same<Derived, CQuadraticTimeMMD>::value && nmethod == N_METHOD::MMD1_GAUSSIAN)
	{
		std::cerr << "cannot use gaussian method for quadratic time MMD" << std::endl;
	}
	else if ((std::is_same<Derived, CBTestMMD>::value || std::is_same<Derived, CLinearTimeMMD>::value) &&
			(nmethod == N_METHOD::MMD2_SPECTRUM || nmethod == N_METHOD::MMD2_GAMMA))
	{
		std::cerr << "cannot use spectrum/gamma method for B-test/linear time MMD" << std::endl;
	}
	self->null_approximation_method = nmethod;
}

template <class Derived>
const N_METHOD CMMD<Derived>::get_null_approximation_method() const
{
	return self->null_approximation_method;
}

template <class Derived>
const char* CMMD<Derived>::get_name() const
{
	return "MMD";
}

template class CMMD<CQuadraticTimeMMD>;
template class CMMD<CLinearTimeMMD>;
template class CMMD<CBTestMMD>;
