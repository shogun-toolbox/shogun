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
#include <shogun/kernel/ShiftInvariantKernel.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/MultiKernelQuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/ComputeMMD.h>
#include <shogun/statistical_testing/internals/mmd/PermutationMMD.h>

using namespace shogun;
using namespace internal;
using namespace mmd;
using std::unique_ptr;

struct CMultiKernelQuadraticTimeMMD::Self
{
	Self(CQuadraticTimeMMD* owner);
	void update_pairwise_distance(CDistance *distance);

	CQuadraticTimeMMD *m_owner;
	unique_ptr<CCustomDistance> m_pairwise_distance;
	EDistanceType m_dtype;
	KernelManager m_kernel_mgr;
	ComputeMMD statistic_job;
	PermutationMMD permutation_job;
};

CMultiKernelQuadraticTimeMMD::Self::Self(CQuadraticTimeMMD *owner) : m_owner(owner),
	m_pairwise_distance(nullptr), m_dtype(D_UNKNOWN)
{
}

void CMultiKernelQuadraticTimeMMD::Self::update_pairwise_distance(CDistance* distance)
{
	ASSERT(distance);
	if (m_dtype==distance->get_distance_type())
	{
		ASSERT(m_pairwise_distance!=nullptr);
		SG_SINFO("Precomputed distance exists for %s!\n", distance->get_name());
	}
	else
	{
		auto precomputed_distance=m_owner->compute_joint_distance(distance);
		m_pairwise_distance=unique_ptr<CCustomDistance>(precomputed_distance);
		m_dtype=distance->get_distance_type();
	}
}

CMultiKernelQuadraticTimeMMD::CMultiKernelQuadraticTimeMMD() : CSGObject()
{
	self=unique_ptr<Self>(new Self(nullptr));
}

CMultiKernelQuadraticTimeMMD::CMultiKernelQuadraticTimeMMD(CQuadraticTimeMMD* owner) : CSGObject()
{
	self=unique_ptr<Self>(new Self(owner));
}

CMultiKernelQuadraticTimeMMD::~CMultiKernelQuadraticTimeMMD()
{
	cleanup();
}

void CMultiKernelQuadraticTimeMMD::add_kernel(CShiftInvariantKernel *kernel)
{
	ASSERT(self->m_owner);
	REQUIRE(kernel, "Kernel instance cannot be NULL!\n");
	self->m_kernel_mgr.push_back(kernel);
}

void CMultiKernelQuadraticTimeMMD::cleanup()
{
	ASSERT(self->m_owner);
	self->m_kernel_mgr.clear();
	self->m_pairwise_distance=nullptr;
	self->m_dtype=D_UNKNOWN;
}

SGVector<float64_t> CMultiKernelQuadraticTimeMMD::compute_statistic()
{
	ASSERT(self->m_owner);
	return statistic(self->m_kernel_mgr);
}

SGVector<float64_t> CMultiKernelQuadraticTimeMMD::compute_variance_h0()
{
	ASSERT(self->m_owner);
	SG_NOTIMPLEMENTED;
	return SGVector<float64_t>();
}

SGVector<float64_t> CMultiKernelQuadraticTimeMMD::compute_variance_h1()
{
	ASSERT(self->m_owner);
	SG_NOTIMPLEMENTED;
	return SGVector<float64_t>();
}

SGMatrix<float32_t> CMultiKernelQuadraticTimeMMD::sample_null()
{
	ASSERT(self->m_owner);
	return sample_null(self->m_kernel_mgr);
}

SGVector<float64_t> CMultiKernelQuadraticTimeMMD::compute_p_value()
{
	ASSERT(self->m_owner);
	return p_values(self->m_kernel_mgr);
}

SGVector<bool> CMultiKernelQuadraticTimeMMD::perform_test(float64_t alpha)
{
	SGVector<float64_t> pvalues=compute_p_value();
	SGVector<bool> rejections(pvalues.size());
	for (auto i=0; i<pvalues.size(); ++i)
	{
		rejections[i]=pvalues[i]<alpha;
	}
	return rejections;
}

SGVector<float64_t> CMultiKernelQuadraticTimeMMD::statistic(const KernelManager& kernel_mgr)
{
	SG_DEBUG("Entering");
	REQUIRE(kernel_mgr.num_kernels()>0, "Number of kernels (%d) have to be greater than 0!\n", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();
	const auto stype = self->m_owner->get_statistic_type();

	CDistance* distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance.get());
	SG_UNREF(distance);

	self->statistic_job.m_n_x=nx;
   	self->statistic_job.m_n_y=ny;
   	self->statistic_job.m_stype=stype;
	SGVector<float64_t> result=self->statistic_job(kernel_mgr);

	kernel_mgr.unset_precomputed_distance();

	for (auto i=0; i<result.vlen; ++i)
		result[i]=self->m_owner->normalize_statistic(result[i]);

	SG_DEBUG("Leaving");
	return result;
}

SGMatrix<float32_t> CMultiKernelQuadraticTimeMMD::sample_null(const KernelManager& kernel_mgr)
{
	SG_DEBUG("Entering");
	REQUIRE(self->m_owner->get_null_approximation_method()==ENullApproximationMethod::PERMUTATION,
		"Multi-kernel tests requires the H0 approximation method to be PERMUTATION!\n");

	REQUIRE(kernel_mgr.num_kernels()>0, "Number of kernels (%d) have to be greater than 0!\n", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();
	const auto stype = self->m_owner->get_statistic_type();
	const auto num_null_samples = self->m_owner->get_num_null_samples();

	CDistance* distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance.get());
	SG_UNREF(distance);

	self->permutation_job.m_n_x=nx;
	self->permutation_job.m_n_y=ny;
   	self->permutation_job.m_num_null_samples=num_null_samples;
	self->permutation_job.m_stype=stype;
	SGMatrix<float32_t> result=self->permutation_job(kernel_mgr);

	kernel_mgr.unset_precomputed_distance();

	for (size_t i=0; i<result.size(); ++i)
		result.matrix[i]=self->m_owner->normalize_statistic(result.matrix[i]);

	SG_DEBUG("Leaving");
	return result;
}

SGVector<float64_t> CMultiKernelQuadraticTimeMMD::p_values(const KernelManager& kernel_mgr)
{
	SG_DEBUG("Entering");
	REQUIRE(self->m_owner->get_null_approximation_method()==ENullApproximationMethod::PERMUTATION,
		"Multi-kernel tests requires the H0 approximation method to be PERMUTATION!\n");

	REQUIRE(kernel_mgr.num_kernels()>0, "Number of kernels (%d) have to be greater than 0!\n", kernel_mgr.num_kernels());

	const auto nx=self->m_owner->get_num_samples_p();
	const auto ny=self->m_owner->get_num_samples_q();
	const auto stype = self->m_owner->get_statistic_type();
	const auto num_null_samples = self->m_owner->get_num_null_samples();

	CDistance* distance=kernel_mgr.get_distance_instance();
	self->update_pairwise_distance(distance);
	kernel_mgr.set_precomputed_distance(self->m_pairwise_distance.get());
	SG_UNREF(distance);

	self->permutation_job.m_n_x=nx;
	self->permutation_job.m_n_y=ny;
   	self->permutation_job.m_num_null_samples=num_null_samples;
	self->permutation_job.m_stype=stype;
	SGVector<float64_t> result=self->permutation_job.p_value(kernel_mgr);

	kernel_mgr.unset_precomputed_distance();

	SG_DEBUG("Leaving");
	return result;
}

const char* CMultiKernelQuadraticTimeMMD::get_name() const
{
	return "MultiKernelQuadraticTimeMMD";
}
