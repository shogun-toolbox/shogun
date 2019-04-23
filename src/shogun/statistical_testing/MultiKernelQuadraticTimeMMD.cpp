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
#include <shogun/kernel/ShiftInvariantKernel.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/MultiKernelQuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/ComputeMMD.h>
#include <shogun/statistical_testing/internals/mmd/VarianceH1.h>
#include <shogun/statistical_testing/internals/mmd/PermutationMMD.h>

using namespace shogun;
using namespace internal;
using namespace mmd;
using std::unique_ptr;

struct MultiKernelQuadraticTimeMMD::Self
{
	Self(QuadraticTimeMMD* owner);
	void update_pairwise_distance(std::shared_ptr<Distance >distance);

	QuadraticTimeMMD* m_owner;
	std::shared_ptr<CustomDistance> m_pairwise_distance;
	EDistanceType m_dtype;
	KernelManager m_kernel_mgr;
	ComputeMMD statistic_job;
	VarianceH1 variance_h1_job;
	PermutationMMD permutation_job;
};

MultiKernelQuadraticTimeMMD::Self::Self(QuadraticTimeMMD* owner) : m_owner(owner),
	m_pairwise_distance(nullptr), m_dtype(D_UNKNOWN)
{
}

void MultiKernelQuadraticTimeMMD::Self::update_pairwise_distance(std::shared_ptr<Distance> distance)
{
	ASSERT(distance);
	if (m_dtype==distance->get_distance_type())
	{
		ASSERT(m_pairwise_distance!=nullptr);
		io::info("Precomputed distance exists for {}!", distance->get_name());
	}
	else
	{
		m_pairwise_distance=m_owner->compute_joint_distance(distance);
		m_dtype=distance->get_distance_type();
	}
}

MultiKernelQuadraticTimeMMD::MultiKernelQuadraticTimeMMD() : RandomMixin<SGObject>()
{
	self=unique_ptr<Self>(new Self(nullptr));
}

MultiKernelQuadraticTimeMMD::MultiKernelQuadraticTimeMMD(QuadraticTimeMMD* owner) : RandomMixin<SGObject>()
{
	self=unique_ptr<Self>(new Self(owner));
}

MultiKernelQuadraticTimeMMD::~MultiKernelQuadraticTimeMMD()
{
	cleanup();
}

void MultiKernelQuadraticTimeMMD::add_kernel(std::shared_ptr<Kernel >kernel)
{
	ASSERT(self->m_owner);
	require(kernel, "Kernel instance cannot be NULL!");
	self->m_kernel_mgr.push_back(kernel->as<ShiftInvariantKernel>());
}

void MultiKernelQuadraticTimeMMD::cleanup()
{
	self->m_kernel_mgr.clear();
	invalidate_precomputed_distance();
}

void MultiKernelQuadraticTimeMMD::invalidate_precomputed_distance()
{
	self->m_pairwise_distance=nullptr;
	self->m_dtype=D_UNKNOWN;
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::compute_statistic()
{
	ASSERT(self->m_owner);
	return statistic(self->m_kernel_mgr);
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::compute_variance_h0()
{
	ASSERT(self->m_owner);
	not_implemented(SOURCE_LOCATION);;
	return SGVector<float64_t>();
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::compute_variance_h1()
{
	ASSERT(self->m_owner);
	return variance_h1(self->m_kernel_mgr);
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::compute_test_power()
{
	ASSERT(self->m_owner);
	return test_power(self->m_kernel_mgr);
}

SGMatrix<float32_t> MultiKernelQuadraticTimeMMD::sample_null()
{
	ASSERT(self->m_owner);
	return sample_null(self->m_kernel_mgr);
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::compute_p_value()
{
	ASSERT(self->m_owner);
	return p_values(self->m_kernel_mgr);
}

SGVector<bool> MultiKernelQuadraticTimeMMD::perform_test(float64_t alpha)
{
	SGVector<float64_t> pvalues=compute_p_value();
	SGVector<bool> rejections(pvalues.size());
	for (auto i=0; i<pvalues.size(); ++i)
	{
		rejections[i]=pvalues[i]<alpha;
	}
	return rejections;
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::statistic(const KernelManager& kernel_mgr)
{
	SG_TRACE("Entering");
	require(kernel_mgr.num_kernels()>0, "Number of kernels ({}) have to be greater than 0!", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();
	const auto stype = self->m_owner->get_statistic_type();

	auto distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance);


	self->statistic_job.m_n_x=nx;
   	self->statistic_job.m_n_y=ny;
   	self->statistic_job.m_stype=stype;
	SGVector<float64_t> result=self->statistic_job(kernel_mgr);

	kernel_mgr.unset_precomputed_distance();

	for (auto i=0; i<result.vlen; ++i)
		result[i]=self->m_owner->normalize_statistic(result[i]);

	SG_TRACE("Leaving");
	return result;
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::variance_h1(const KernelManager& kernel_mgr)
{
	SG_TRACE("Entering");
	require(kernel_mgr.num_kernels()>0, "Number of kernels ({}) have to be greater than 0!", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();

	auto distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance);


	self->variance_h1_job.m_n_x=nx;
   	self->variance_h1_job.m_n_y=ny;
	SGVector<float64_t> result=self->variance_h1_job(kernel_mgr);

	kernel_mgr.unset_precomputed_distance();

	SG_TRACE("Leaving");
	return result;
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::test_power(const KernelManager& kernel_mgr)
{
	SG_TRACE("Entering");
	require(kernel_mgr.num_kernels()>0, "Number of kernels ({}) have to be greater than 0!", kernel_mgr.num_kernels());
	require(self->m_owner->get_statistic_type()==ST_UNBIASED_FULL, "Only possible with UNBIASED_FULL!");

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();

	auto distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance);


	self->variance_h1_job.m_n_x=nx;
   	self->variance_h1_job.m_n_y=ny;
	SGVector<float64_t> result=self->variance_h1_job.test_power(kernel_mgr);

	kernel_mgr.unset_precomputed_distance();

	SG_TRACE("Leaving");
	return result;
}

SGMatrix<float32_t> MultiKernelQuadraticTimeMMD::sample_null(const KernelManager& kernel_mgr)
{
	SG_TRACE("Entering");
	require(self->m_owner->get_null_approximation_method()==NAM_PERMUTATION,
		"Multi-kernel tests requires the H0 approximation method to be PERMUTATION!");

	require(kernel_mgr.num_kernels()>0, "Number of kernels ({}) have to be greater than 0!", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();
	const auto stype = self->m_owner->get_statistic_type();
	const auto num_null_samples = self->m_owner->get_num_null_samples();

	auto distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance);


	self->permutation_job.m_n_x=nx;
	self->permutation_job.m_n_y=ny;
   	self->permutation_job.m_num_null_samples=num_null_samples;
	self->permutation_job.m_stype=stype;
	SGMatrix<float32_t> result=self->permutation_job(kernel_mgr, m_prng);

	kernel_mgr.unset_precomputed_distance();

	for (index_t i=0; i<result.size(); ++i)
		result.matrix[i]=self->m_owner->normalize_statistic(result.matrix[i]);

	SG_TRACE("Leaving");
	return result;
}

SGVector<float64_t> MultiKernelQuadraticTimeMMD::p_values(const KernelManager& kernel_mgr)
{
	SG_TRACE("Entering");
	require(self->m_owner->get_null_approximation_method()==NAM_PERMUTATION,
		"Multi-kernel tests requires the H0 approximation method to be PERMUTATION!");

	require(kernel_mgr.num_kernels()>0, "Number of kernels ({}) have to be greater than 0!", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();
	const auto stype = self->m_owner->get_statistic_type();
	const auto num_null_samples = self->m_owner->get_num_null_samples();

	auto distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance);


	self->permutation_job.m_n_x=nx;
	self->permutation_job.m_n_y=ny;
   	self->permutation_job.m_num_null_samples=num_null_samples;
	self->permutation_job.m_stype=stype;
	SGVector<float64_t> result=self->permutation_job.p_value(kernel_mgr, m_prng);

	kernel_mgr.unset_precomputed_distance();

	SG_TRACE("Leaving");
	return result;
}

const char* MultiKernelQuadraticTimeMMD::get_name() const
{
	return "MultiKernelQuadraticTimeMMD";
}
