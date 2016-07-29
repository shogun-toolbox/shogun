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
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/BTestMMD.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/ComputationManager.h>
#include <shogun/statistical_testing/internals/mmd/ComputeMMD.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockPermutation.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace internal;

struct CStreamingMMD::Self
{
	Self(CStreamingMMD& cmmd);

	void create_statistic_job();
	void create_variance_job();
	void create_computation_jobs();

	void merge_samples(NextSamples&, std::vector<CFeatures*>&) const;
	void compute_kernel(ComputationManager&, std::vector<CFeatures*>&, CKernel*) const;
	void compute_jobs(ComputationManager&) const;

	std::pair<float64_t, float64_t> compute_statistic_variance();
	std::pair<SGVector<float64_t>, SGMatrix<float64_t>> compute_statistic_and_Q(const KernelManager&);
	SGVector<float64_t> sample_null();

	CStreamingMMD& owner;

	bool use_gpu;
	index_t num_null_samples;

	EStatisticType statistic_type;
	EVarianceEstimationMethod variance_estimation_method;
	ENullApproximationMethod null_approximation_method;

	std::function<float32_t(const SGMatrix<float32_t>&)> statistic_job;
	std::function<float32_t(const SGMatrix<float32_t>&)> permutation_job;
	std::function<float32_t(const SGMatrix<float32_t>&)> variance_job;
};

CStreamingMMD::Self::Self(CStreamingMMD& cmmd) : owner(cmmd),
	use_gpu(false), num_null_samples(250),
	statistic_type(ST_UNBIASED_FULL),
	variance_estimation_method(VEM_DIRECT),
	null_approximation_method(NAM_PERMUTATION),
	statistic_job(nullptr), variance_job(nullptr)
{
}

void CStreamingMMD::Self::create_computation_jobs()
{
	create_statistic_job();
	create_variance_job();
}

void CStreamingMMD::Self::create_statistic_job()
{
	const DataManager& data_mgr=owner.get_data_mgr();

	auto Bx=data_mgr.blocksize_at(0);
	auto By=data_mgr.blocksize_at(1);

	REQUIRE(Bx>0, "Blocksize for samples from P cannot be 0!\n");
	REQUIRE(By>0, "Blocksize for samples from Q cannot be 0!\n");

	auto mmd=mmd::ComputeMMD();
	mmd.m_n_x=Bx;
	mmd.m_n_y=By;
	mmd.m_stype=statistic_type;

	statistic_job=mmd;
	permutation_job=mmd::WithinBlockPermutation(Bx, By, statistic_type);
}

void CStreamingMMD::Self::create_variance_job()
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

void CStreamingMMD::Self::merge_samples(NextSamples& next_burst, std::vector<CFeatures*>& blocks) const
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

void CStreamingMMD::Self::compute_kernel(ComputationManager& cm, std::vector<CFeatures*>& blocks, CKernel* kernel) const
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

void CStreamingMMD::Self::compute_jobs(ComputationManager& cm) const
{
	if (use_gpu)
		cm.use_gpu().compute_data_parallel_jobs();
	else
		cm.use_cpu().compute_data_parallel_jobs();
}

std::pair<float64_t, float64_t> CStreamingMMD::Self::compute_statistic_variance()
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
	data_mgr.start();
	auto next_burst=data_mgr.next();
	if (!next_burst.empty())
	{
		ComputationManager cm;
		create_computation_jobs();
		cm.enqueue_job(statistic_job);
		cm.enqueue_job(variance_job);

		std::vector<CFeatures*> blocks;

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
		cm.done();
	}
	data_mgr.end();

	// normalize statistic and variance
	statistic=owner.normalize_statistic(statistic);
	if (variance_estimation_method==VEM_PERMUTATION)
		variance=owner.normalize_variance(variance);

	return std::make_pair(statistic, variance);
}

std::pair<SGVector<float64_t>, SGMatrix<float64_t> > CStreamingMMD::Self::compute_statistic_and_Q(const KernelManager& kernel_selection_mgr)
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

SGVector<float64_t> CStreamingMMD::Self::sample_null()
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

CStreamingMMD::CStreamingMMD() : CMMD()
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	self=std::unique_ptr<Self>(new Self(*this));
}

CStreamingMMD::~CStreamingMMD()
{
}

float64_t CStreamingMMD::compute_statistic()
{
	return self->compute_statistic_variance().first;
}

float64_t CStreamingMMD::compute_variance()
{
	return self->compute_statistic_variance().second;
}

SGVector<float64_t> CStreamingMMD::compute_multiple()
{
	return self->compute_statistic_and_Q(get_kernel_selection_strategy()->get_kernel_mgr()).first;
}

std::pair<float64_t, float64_t> CStreamingMMD::compute_statistic_variance()
{
	return self->compute_statistic_variance();
}

std::pair<SGVector<float64_t>, SGMatrix<float64_t> > CStreamingMMD::compute_statistic_and_Q(const KernelManager& kernel_selection_mgr)
{
	return self->compute_statistic_and_Q(kernel_selection_mgr);
}

SGVector<float64_t> CStreamingMMD::sample_null()
{
	return self->sample_null();
}

void CStreamingMMD::set_num_null_samples(index_t null_samples)
{
	self->num_null_samples=null_samples;
}

const index_t CStreamingMMD::get_num_null_samples() const
{
	return self->num_null_samples;
}

void CStreamingMMD::use_gpu(bool gpu)
{
	self->use_gpu=gpu;
}

bool CStreamingMMD::use_gpu() const
{
	return self->use_gpu;
}

void CStreamingMMD::cleanup()
{
	for (size_t i=0; i<get_kernel_mgr().num_kernels(); ++i)
		get_kernel_mgr().restore_kernel_at(i);
}

void CStreamingMMD::set_statistic_type(EStatisticType stype)
{
	self->statistic_type=stype;
}

const EStatisticType CStreamingMMD::get_statistic_type() const
{
	return self->statistic_type;
}

void CStreamingMMD::set_variance_estimation_method(EVarianceEstimationMethod vmethod)
{
	// TODO overload this
/*	if (std::is_same<Derived, CQuadraticTimeMMD>::value && vmethod == VEM_PERMUTATION)
	{
		std::cerr << "cannot use permutation method for quadratic time MMD" << std::endl;
	}*/
	self->variance_estimation_method=vmethod;
}

const EVarianceEstimationMethod CStreamingMMD::get_variance_estimation_method() const
{
	return self->variance_estimation_method;
}

void CStreamingMMD::set_null_approximation_method(ENullApproximationMethod nmethod)
{
	// TODO overload this
/*	if (std::is_same<Derived, CQuadraticTimeMMD>::value && nmethod == NAM_MMD1_GAUSSIAN)
	{
		std::cerr << "cannot use gaussian method for quadratic time MMD" << std::endl;
	}
	else if ((std::is_same<Derived, CBTestMMD>::value || std::is_same<Derived, CLinearTimeMMD>::value) &&
			(nmethod == NAM_MMD2_SPECTRUM || nmethod == NAM_MMD2_GAMMA))
	{
		std::cerr << "cannot use spectrum/gamma method for B-test/linear time MMD" << std::endl;
	}*/
	self->null_approximation_method=nmethod;
}

const ENullApproximationMethod CStreamingMMD::get_null_approximation_method() const
{
	return self->null_approximation_method;
}

const char* CStreamingMMD::get_name() const
{
	return "StreamingMMD";
}
