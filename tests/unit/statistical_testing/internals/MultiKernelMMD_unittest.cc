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

#include <shogun/base/some.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/statistical_testing/TwoDistributionTest.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/MultiKernelMMD.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace shogun
{

class CTwoDistributionTestMock : public CTwoDistributionTest
{
public:
	MOCK_METHOD0(compute_statistic, float64_t());
	MOCK_METHOD0(sample_null, SGVector<float64_t>());
};

}

using namespace shogun;
using namespace internal;
using namespace mmd;
using Eigen::Map;
using Eigen::MatrixXd;

TEST(MultiKernelMMD, biased_full)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;
	const EStatisticType stype=ST_BIASED_FULL;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(m);
	auto feats_q=gen_q->get_streamed_features(n);

	KernelManager kernel_mgr(num_kernels);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
		kernel_mgr.kernel_at(i)=new CGaussianKernel(10, pow(2, sigma));

	auto test=some<CTwoDistributionTestMock>();
	test->set_p(feats_p);
	test->set_q(feats_q);

	MultiKernelMMD tester(m, n, stype);
	tester.set_distance(test->compute_distance());
	SGVector<float64_t> values=tester(kernel_mgr);

	auto data_p=static_cast<CDenseFeatures<float64_t>*>(feats_p)->get_feature_matrix();
	auto data_q=static_cast<CDenseFeatures<float64_t>*>(feats_q)->get_feature_matrix();
	SGMatrix<float64_t> data_p_and_q(dim, m+n);
	std::copy(data_p.data(), data_p.data()+data_p.size(), data_p_and_q.data());
	std::copy(data_q.data(), data_q.data()+data_q.size(), data_p_and_q.data()+data_p.size());
	auto feats_p_and_q=new CDenseFeatures<float64_t>(data_p_and_q);
	SG_REF(feats_p_and_q);

	SGVector<float64_t> ref(kernel_mgr.num_kernels());
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
	{
		CKernel* kernel=kernel_mgr.kernel_at(i);
		kernel->init(feats_p_and_q, feats_p_and_q);
		SGMatrix<float64_t> km=kernel->get_kernel_matrix();
		Map<MatrixXd> map(km.data(), km.num_rows, km.num_cols);
		auto term_0=map.block(0, 0, m, m).sum();
		auto term_1=map.block(m, m, n, n).sum();
		auto term_2=map.block(m, 0, n, m).sum();
		term_0/=m*m;
		term_1/=n*n;
		term_2/=m*n;
		ref[i]=term_0+term_1-2*term_2;
		kernel->remove_lhs_and_rhs();
	}
	SG_UNREF(feats_p_and_q);

	ASSERT_EQ(ref.size(), values.size());
	for (auto i=0; i<ref.size(); ++i)
	{
		EXPECT_NEAR(ref[i], values[i], 1E-6);
	}
}

TEST(MultiKernelMMD, unbiased_full)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;
	const EStatisticType stype=ST_UNBIASED_FULL;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(m);
	auto feats_q=gen_q->get_streamed_features(n);

	KernelManager kernel_mgr(num_kernels);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
		kernel_mgr.kernel_at(i)=new CGaussianKernel(10, pow(2, sigma));

	auto test=some<CTwoDistributionTestMock>();
	test->set_p(feats_p);
	test->set_q(feats_q);

	MultiKernelMMD tester(m, n, stype);
	tester.set_distance(test->compute_distance());
	SGVector<float64_t> values=tester(kernel_mgr);

	auto data_p=static_cast<CDenseFeatures<float64_t>*>(feats_p)->get_feature_matrix();
	auto data_q=static_cast<CDenseFeatures<float64_t>*>(feats_q)->get_feature_matrix();
	SGMatrix<float64_t> data_p_and_q(dim, m+n);
	std::copy(data_p.data(), data_p.data()+data_p.size(), data_p_and_q.data());
	std::copy(data_q.data(), data_q.data()+data_q.size(), data_p_and_q.data()+data_p.size());
	auto feats_p_and_q=new CDenseFeatures<float64_t>(data_p_and_q);
	SG_REF(feats_p_and_q);

	SGVector<float64_t> ref(kernel_mgr.num_kernels());
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
	{
		CKernel* kernel=kernel_mgr.kernel_at(i);
		kernel->init(feats_p_and_q, feats_p_and_q);
		SGMatrix<float64_t> km=kernel->get_kernel_matrix();
		Map<MatrixXd> map(km.data(), km.num_rows, km.num_cols);
		auto term_0=map.block(0, 0, m, m).sum()-map.diagonal().head(m).sum();
		auto term_1=map.block(m, m, n, n).sum()-map.diagonal().tail(n).sum();
		auto term_2=map.block(m, 0, n, m).sum();
		term_0/=m*(m-1);
		term_1/=n*(n-1);
		term_2/=m*n;
		ref[i]=term_0+term_1-2*term_2;
		kernel->remove_lhs_and_rhs();
	}
	SG_UNREF(feats_p_and_q);

	ASSERT_EQ(ref.size(), values.size());
	for (auto i=0; i<ref.size(); ++i)
	{
		EXPECT_NEAR(ref[i], values[i], 1E-6);
	}
}

TEST(MultiKernelMMD, unbiased_incomplete)
{
	const index_t m=8;
	const index_t n=8;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=1;
	const EStatisticType stype=ST_UNBIASED_INCOMPLETE;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(m);
	auto feats_q=gen_q->get_streamed_features(n);

	KernelManager kernel_mgr(num_kernels);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
		kernel_mgr.kernel_at(i)=new CGaussianKernel(10, pow(2, sigma));

	auto test=some<CTwoDistributionTestMock>();
	test->set_p(feats_p);
	test->set_q(feats_q);

	MultiKernelMMD tester(m, n, stype);
	tester.set_distance(test->compute_distance());
	SGVector<float64_t> values=tester(kernel_mgr);

	auto data_p=static_cast<CDenseFeatures<float64_t>*>(feats_p)->get_feature_matrix();
	auto data_q=static_cast<CDenseFeatures<float64_t>*>(feats_q)->get_feature_matrix();
	SGMatrix<float64_t> data_p_and_q(dim, m+n);
	std::copy(data_p.data(), data_p.data()+data_p.size(), data_p_and_q.data());
	std::copy(data_q.data(), data_q.data()+data_q.size(), data_p_and_q.data()+data_p.size());
	auto feats_p_and_q=new CDenseFeatures<float64_t>(data_p_and_q);
	SG_REF(feats_p_and_q);

	SGVector<float64_t> ref(kernel_mgr.num_kernels());
	for (size_t i=0; i<kernel_mgr.num_kernels(); ++i)
	{
		CKernel* kernel=kernel_mgr.kernel_at(i);
		kernel->init(feats_p_and_q, feats_p_and_q);
		SGMatrix<float64_t> km=kernel->get_kernel_matrix();
		Map<MatrixXd> map(km.data(), km.num_rows, km.num_cols);
		auto term_0=map.block(0, 0, m, m).sum()-map.diagonal().head(m).sum();
		auto term_1=map.block(m, m, n, n).sum()-map.diagonal().tail(n).sum();
		auto term_2=map.block(m, 0, n, m).sum()-map.block(m, 0, n, m).diagonal().sum();
		term_0/=m*(m-1);
		term_1/=n*(n-1);
		term_2/=m*(n-1);
		ref[i]=term_0+term_1-2*term_2;
		kernel->remove_lhs_and_rhs();
	}
	SG_UNREF(feats_p_and_q);

	ASSERT_EQ(ref.size(), values.size());
	for (auto i=0; i<ref.size(); ++i)
	{
		EXPECT_NEAR(ref[i], values[i], 1E-6);
	}
}
