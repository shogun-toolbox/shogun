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
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/BTestMMD.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/KernelSelectionStrategy.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/ComputationManager.h>
#include <shogun/statistical_testing/internals/MaxMeasure.h>
#include <shogun/statistical_testing/internals/MaxTestPower.h>
#include <shogun/statistical_testing/internals/MaxXValidation.h>
#include <shogun/statistical_testing/internals/MedianHeuristic.h>
#include <shogun/statistical_testing/internals/WeightedMaxMeasure.h>
#include <shogun/statistical_testing/internals/WeightedMaxTestPower.h>
#include <shogun/statistical_testing/internals/mmd/BiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedIncomplete.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockPermutation.h>
#include <shogun/mathematics/eigen3.h>

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
	std::pair<SGVector<float64_t>, SGMatrix<float64_t>> compute_statistic_and_Q(const KernelManager&);
	CCustomDistance* compute_distance();
	SGVector<float64_t> sample_null();

	CMMD& owner;

	bool use_gpu;
	index_t num_null_samples;

	EStatisticType statistic_type;
	EVarianceEstimationMethod variance_estimation_method;
	ENullApproximationMethod null_approximation_method;

	std::function<float32_t(const SGMatrix<float32_t>&)> statistic_job;
	std::function<float32_t(const SGMatrix<float32_t>&)> permutation_job;
	std::function<float32_t(const SGMatrix<float32_t>&)> variance_job;

	std::shared_ptr<CKernelSelectionStrategy> strategy;
};

CMMD::Self::Self(CMMD& cmmd) : owner(cmmd),
	use_gpu(false), num_null_samples(250),
	statistic_type(ST_UNBIASED_FULL),
	variance_estimation_method(VEM_DIRECT),
	null_approximation_method(NAM_PERMUTATION),
	statistic_job(nullptr), variance_job(nullptr)
{
	auto default_strategy=new CKernelSelectionStrategy();
	SG_REF(default_strategy);
	strategy=std::shared_ptr<CKernelSelectionStrategy>(default_strategy, [](CKernelSelectionStrategy* ptr) { SG_UNREF(ptr); });
}

void CMMD::Self::create_computation_jobs()
{
	create_statistic_job();
	create_variance_job();
}

void CMMD::Self::create_statistic_job()
{
	const DataManager& data_mgr=owner.get_data_mgr();
	auto Bx=data_mgr.blocksize_at(0);
	auto By=data_mgr.blocksize_at(1);
	switch (statistic_type)
	{
		case ST_UNBIASED_FULL:
			statistic_job=mmd::UnbiasedFull(Bx);
			break;
		case ST_UNBIASED_INCOMPLETE:
			statistic_job=mmd::UnbiasedIncomplete(Bx);
			break;
		case ST_BIASED_FULL:
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
		case VEM_DIRECT:
			variance_job=owner.get_direct_estimation_method();
			break;
		case VEM_PERMUTATION:
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
	const KernelManager& kernel_mgr=owner.get_kernel_mgr();
	auto kernel=kernel_mgr.kernel_at(0);
	REQUIRE(kernel != nullptr, "Kernel is not set!\n");

	float64_t statistic=0;
	float64_t permuted_samples_statistic=0;
	float64_t variance=0;
	index_t statistic_term_counter=1;
	index_t variance_term_counter=1;

	DataManager& data_mgr=owner.get_data_mgr();
	ComputationManager cm;

	create_computation_jobs();
	cm.enqueue_job(statistic_job);
	cm.enqueue_job(variance_job);

	std::vector<CFeatures*> blocks;

	data_mgr.start();
	auto next_burst=data_mgr.next();
	while (!next_burst.empty())
	{
		merge_samples(next_burst, blocks);
		compute_kernel(cm, blocks, kernel);
		blocks.resize(0);
		compute_jobs(cm);

		auto mmds=cm.result(0);
		auto vars=cm.result(1);

		for (size_t i=0; i<mmds.size(); ++i)
		{
			auto delta=mmds[i]-statistic;
			statistic+=delta/statistic_term_counter;
			statistic_term_counter++;
		}

		if (variance_estimation_method==VEM_DIRECT)
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
		next_burst=data_mgr.next();
	}

	data_mgr.end();
	cm.done();

	// normalize statistic and variance
	statistic=owner.normalize_statistic(statistic);
	if (variance_estimation_method==VEM_PERMUTATION)
		variance=owner.normalize_variance(variance);

	return std::make_pair(statistic, variance);
}

std::pair<SGVector<float64_t>, SGMatrix<float64_t> > CMMD::Self::compute_statistic_and_Q(const KernelManager& kernel_selection_mgr)
{
//	const size_t num_kernels=0;
//	SGVector<float64_t> statistic(num_kernels);
//	SGMatrix<float64_t> Q(num_kernels, num_kernels);
//	return std::make_pair(statistic, Q);
	REQUIRE(kernel_selection_mgr.num_kernels()>0, "No kernels specified for kernel learning! "
		"Please add kernels using add_kernel() method!\n");

	const size_t num_kernels=kernel_selection_mgr.num_kernels();
	SGVector<float64_t> statistic(num_kernels);
	SGMatrix<float64_t> Q(num_kernels, num_kernels);

	std::fill(statistic.data(), statistic.data()+statistic.size(), 0);
	std::fill(Q.data(), Q.data()+Q.size(), 0);

	std::vector<index_t> term_counters_statistic(num_kernels, 1);
	SGMatrix<index_t> term_counters_Q(num_kernels, num_kernels);
	std::fill(term_counters_Q.data(), term_counters_Q.data()+term_counters_Q.size(), 1);

	DataManager& data_mgr=owner.get_data_mgr();
	ComputationManager cm;
	create_computation_jobs();
	cm.enqueue_job(statistic_job);

	data_mgr.start();
	auto next_burst=data_mgr.next();
	std::vector<CFeatures*> blocks;
	std::vector<std::vector<float32_t> > mmds(num_kernels);
	while (!next_burst.empty())
	{
		const size_t num_blocks=next_burst.num_blocks();
		REQUIRE(num_blocks%2==0,
				"The number of blocks per burst (%d this burst) has to be even!\n",
				num_blocks);
		merge_samples(next_burst, blocks);
		std::for_each(blocks.begin(), blocks.end(), [](CFeatures* ptr) { SG_REF(ptr); });
		for (size_t k=0; k<num_kernels; ++k)
		{
			CKernel* kernel=kernel_selection_mgr.kernel_at(k);
			compute_kernel(cm, blocks, kernel);
			compute_jobs(cm);
			mmds[k]=cm.result(0);
			for (size_t i=0; i<num_blocks; ++i)
			{
				auto delta=mmds[k][i]-statistic[k];
				statistic[k]+=delta/term_counters_statistic[k]++;
			}
		}
		std::for_each(blocks.begin(), blocks.end(), [](CFeatures* ptr) { SG_UNREF(ptr); });
		blocks.resize(0);
		for (size_t i=0; i<num_kernels; ++i)
		{
			for (size_t j=0; j<=i; ++j)
			{
				for (size_t k=0; k<num_blocks-1; k+=2)
				{
					auto term=(mmds[i][k]-mmds[i][k+1])*(mmds[j][k]-mmds[j][k+1]);
					Q(i, j)+=(term-Q(i, j))/term_counters_Q(i, j)++;
				}
				Q(j, i)=Q(i, j);
			}
		}
		next_burst=data_mgr.next();
	}
	mmds.clear();

	data_mgr.end();
	cm.done();

	std::for_each(statistic.data(), statistic.data()+statistic.size(), [this](float64_t val)
	{
		val=owner.normalize_statistic(val);
	});
	return std::make_pair(statistic, Q);
}

SGVector<float64_t> CMMD::Self::sample_null()
{
	const KernelManager& kernel_mgr=owner.get_kernel_mgr();
	auto kernel=kernel_mgr.kernel_at(0);
	REQUIRE(kernel != nullptr, "Kernel is not set!\n");

	SGVector<float64_t> statistic(num_null_samples);
	std::vector<index_t> term_counters(num_null_samples);

	std::fill(statistic.vector, statistic.vector+statistic.vlen, 0);
	std::fill(term_counters.data(), term_counters.data()+term_counters.size(), 1);

	DataManager& data_mgr=owner.get_data_mgr();
	ComputationManager cm;

	create_statistic_job();
	cm.enqueue_job(permutation_job);

	std::vector<CFeatures*> blocks;

	data_mgr.start();
	auto next_burst=data_mgr.next();

	while (!next_burst.empty())
	{
		merge_samples(next_burst, blocks);
		compute_kernel(cm, blocks, kernel);
		blocks.resize(0);

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
		next_burst=data_mgr.next();
	}

	data_mgr.end();
	cm.done();

	// normalize statistic
	std::for_each(statistic.vector, statistic.vector + statistic.vlen, [this](float64_t& value)
	{
		value=owner.normalize_statistic(value);
	});

	return statistic;
}

CCustomDistance* CMMD::Self::compute_distance()
{
	auto distance=new CCustomDistance();
	DataManager& data_mgr=owner.get_data_mgr();

	bool blockwise=data_mgr.is_blockwise();
	data_mgr.set_blockwise(false);

	// using data manager next() API in order to make it work with
	// streaming samples as well.
	data_mgr.start();
	auto samples=data_mgr.next();
	if (!samples.empty())
	{
		// use 0th block from each distribution (since there is only one block
		// for quadratic time MMD
		CFeatures *samples_p=samples[0][0].get();
		CFeatures *samples_q=samples[1][0].get();

		try
		{
			auto p_and_q=FeaturesUtil::create_merged_copy(samples_p, samples_q);
			samples.clear();
			auto euclidean_distance=std::unique_ptr<CEuclideanDistance>(new CEuclideanDistance());
			if (euclidean_distance->init(p_and_q, p_and_q))
			{
				auto dist_mat=euclidean_distance->get_distance_matrix<float32_t>();
				distance->set_triangle_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
			}
			else
			{
				SG_SERROR("Computing distance matrix was not possible! Please contact Shogun developers.\n");
			}
		}
		catch (ShogunException e)
		{
			SG_SERROR("%s, Data is too large! Computing distance matrix was not possible!\n", e.get_exception_string());
		}
	}
	else
		SG_SERROR("Could not fetch samples!\n");

	data_mgr.end();
	data_mgr.set_blockwise(blockwise);

	return distance;
}

CMMD::CMMD() : CTwoSampleTest()
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	self=std::unique_ptr<Self>(new Self(*this));
}

CMMD::~CMMD()
{
}

void CMMD::set_kernel_selection_strategy(EKernelSelectionMethod method)
{
	auto strategy=std::shared_ptr<CKernelSelectionStrategy>(new CKernelSelectionStrategy(method));
	const auto& kernel_mgr=self->strategy->get_kernel_mgr();
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
		strategy->add_kernel(kernel_mgr.kernel_at(i));
	self->strategy=strategy;
}

void CMMD::set_kernel_selection_strategy(EKernelSelectionMethod method, bool weighted)
{
	auto strategy=std::shared_ptr<CKernelSelectionStrategy>(new CKernelSelectionStrategy(method, weighted));
	const auto& kernel_mgr=self->strategy->get_kernel_mgr();
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
		strategy->add_kernel(kernel_mgr.kernel_at(i));
	self->strategy=strategy;
}

void CMMD::set_kernel_selection_strategy(EKernelSelectionMethod method, index_t num_runs, float64_t alpha)
{
	auto strategy=std::shared_ptr<CKernelSelectionStrategy>(new CKernelSelectionStrategy(method, num_runs, alpha));
	const auto& kernel_mgr=self->strategy->get_kernel_mgr();
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
		strategy->add_kernel(kernel_mgr.kernel_at(i));
	self->strategy=strategy;
}

CKernelSelectionStrategy* CMMD::get_kernel_selection_strategy() const
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

	auto& kernel_mgr=get_kernel_mgr();
	kernel_mgr.kernel_at(0)=self->strategy->select_kernel(this);
	kernel_mgr.restore_kernel_at(0);

	data_mgr.set_train_mode(false);
	SG_DEBUG("Leaving!\n");
}

CCustomDistance* CMMD::compute_distance()
{
	return self->compute_distance();
}

float64_t CMMD::compute_statistic()
{
	return self->compute_statistic_variance().first;
}

float64_t CMMD::compute_variance()
{
	return self->compute_statistic_variance().second;
}

void CMMD::set_train_test_ratio(float64_t ratio)
{
	auto& data_mgr=get_data_mgr();
	data_mgr.set_train_test_ratio(ratio);
	data_mgr.reset();
}

std::pair<float64_t, float64_t> CMMD::compute_statistic_variance()
{
	return self->compute_statistic_variance();
}

std::pair<SGVector<float64_t>, SGMatrix<float64_t> > CMMD::compute_statistic_and_Q(const KernelManager& kernel_selection_mgr)
{
	return self->compute_statistic_and_Q(kernel_selection_mgr);
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
	for (size_t i=0; i<get_kernel_mgr().num_kernels(); ++i)
		get_kernel_mgr().restore_kernel_at(i);
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
