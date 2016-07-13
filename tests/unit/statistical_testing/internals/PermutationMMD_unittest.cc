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

#include <algorithm>
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
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/internals/Kernel.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/mmd/ComputeMMD.h>
#include <shogun/statistical_testing/internals/mmd/PermutationMMD.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::Dynamic;
using Eigen::PermutationMatrix;

TEST(PermutationMMD, biased_full_single_kernel)
{
	const index_t dim=2;
	const index_t n=13;
	const index_t m=7;
	const index_t num_null_samples=5;
	const auto stype=EStatisticType::BIASED_FULL;

	SGMatrix<float64_t> data_p(dim, n);
	std::iota(data_p.matrix, data_p.matrix+dim*n, 1);
	std::for_each(data_p.matrix, data_p.matrix+dim*n, [&n](float64_t& val) { val/=n; });

	SGMatrix<float64_t> data_q(dim, m);
	std::iota(data_q.matrix, data_q.matrix+dim*m, n+1);
	std::for_each(data_q.matrix, data_q.matrix+dim*m, [&m](float64_t& val) { val/=2*m; });

	auto feats_p=new CDenseFeatures<float64_t>(data_p);
	auto feats_q=new CDenseFeatures<float64_t>(data_q);
	auto feats=feats_p->create_merged_copy(feats_q);
	SG_REF(feats);
	SG_UNREF(feats_p);
	SG_UNREF(feats_q);

	auto kernel=some<CGaussianKernel>();
	kernel->set_width(2.0);

	kernel->init(feats, feats);
	auto kernel_matrix=kernel->get_kernel_matrix<float32_t>();

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=m;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGVector<float32_t> result_1=permutation_mmd(kernel_matrix);

	auto compute_mmd=ComputeMMD();
	compute_mmd.m_n_x=n;
	compute_mmd.m_n_y=m;
	compute_mmd.m_stype=stype;

	Map<MatrixXf> map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);
	SGVector<float32_t> result_2(num_null_samples);
	sg_rand->set_seed(12345);
	for (auto i=0; i<num_null_samples; ++i)
	{
		PermutationMatrix<Dynamic, Dynamic> perm(kernel_matrix.num_rows);
		perm.setIdentity();
		SGVector<int> perminds(perm.indices().data(), perm.indices().size(), false);
		CMath::permute(perminds);
		MatrixXf permuted = perm.transpose()*map*perm;
		SGMatrix<float32_t> permuted_km(permuted.data(), permuted.rows(), permuted.cols(), false);
		result_2[i]=compute_mmd(permuted_km);
	}

	SGVector<index_t> inds(kernel_matrix.num_rows);
	SGVector<float32_t> result_3(num_null_samples);
	sg_rand->set_seed(12345);
	for (auto i=0; i<num_null_samples; ++i)
	{
		std::iota(inds.vector, inds.vector+inds.vlen, 0);
		CMath::permute(inds);
		feats->add_subset(inds);
		kernel->init(feats, feats);
		kernel_matrix=kernel->get_kernel_matrix<float32_t>();
		result_3[i]=compute_mmd(kernel_matrix);
		feats->remove_subset();
	}

	for (auto i=0; i<num_null_samples; ++i)
	{
		EXPECT_NEAR(result_1[i], result_2[i], 1E-6);
		EXPECT_NEAR(result_1[i], result_3[i], 1E-6);
	}

	SG_UNREF(feats);
}

TEST(PermutationMMD, unbiased_full_single_kernel)
{
	const index_t dim=2;
	const index_t n=13;
	const index_t m=7;
	const index_t num_null_samples=5;
	const auto stype=EStatisticType::UNBIASED_FULL;

	SGMatrix<float64_t> data_p(dim, n);
	std::iota(data_p.matrix, data_p.matrix+dim*n, 1);
	std::for_each(data_p.matrix, data_p.matrix+dim*n, [&n](float64_t& val) { val/=n; });

	SGMatrix<float64_t> data_q(dim, m);
	std::iota(data_q.matrix, data_q.matrix+dim*m, n+1);
	std::for_each(data_q.matrix, data_q.matrix+dim*m, [&m](float64_t& val) { val/=2*m; });

	auto feats_p=new CDenseFeatures<float64_t>(data_p);
	auto feats_q=new CDenseFeatures<float64_t>(data_q);
	auto feats=feats_p->create_merged_copy(feats_q);
	SG_REF(feats);
	SG_UNREF(feats_p);
	SG_UNREF(feats_q);

	auto kernel=some<CGaussianKernel>();
	kernel->set_width(2.0);

	kernel->init(feats, feats);
	auto kernel_matrix=kernel->get_kernel_matrix<float32_t>();

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=m;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGVector<float32_t> result_1=permutation_mmd(kernel_matrix);

	auto compute_mmd=ComputeMMD();
	compute_mmd.m_n_x=n;
	compute_mmd.m_n_y=m;
	compute_mmd.m_stype=stype;

	Map<MatrixXf> map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);
	SGVector<float32_t> result_2(num_null_samples);
	sg_rand->set_seed(12345);
	for (auto i=0; i<num_null_samples; ++i)
	{
		PermutationMatrix<Dynamic, Dynamic> perm(kernel_matrix.num_rows);
		perm.setIdentity();
		SGVector<int> perminds(perm.indices().data(), perm.indices().size(), false);
		CMath::permute(perminds);
		MatrixXf permuted = perm.transpose()*map*perm;
		SGMatrix<float32_t> permuted_km(permuted.data(), permuted.rows(), permuted.cols(), false);
		result_2[i]=compute_mmd(permuted_km);
	}

	SGVector<index_t> inds(kernel_matrix.num_rows);
	SGVector<float32_t> result_3(num_null_samples);
	sg_rand->set_seed(12345);
	for (auto i=0; i<num_null_samples; ++i)
	{
		std::iota(inds.vector, inds.vector+inds.vlen, 0);
		CMath::permute(inds);
		feats->add_subset(inds);
		kernel->init(feats, feats);
		kernel_matrix=kernel->get_kernel_matrix<float32_t>();
		result_3[i]=compute_mmd(kernel_matrix);
		feats->remove_subset();
	}

	for (auto i=0; i<num_null_samples; ++i)
	{
		EXPECT_NEAR(result_1[i], result_2[i], 1E-6);
		EXPECT_NEAR(result_1[i], result_3[i], 1E-6);
	}

	SG_UNREF(feats);
}

TEST(PermutationMMD, unbiased_incomplete_single_kernel)
{
	const index_t dim=2;
	const index_t n=10;
	const index_t num_null_samples=5;
	const auto stype=EStatisticType::UNBIASED_INCOMPLETE;

	SGMatrix<float64_t> data_p(dim, n);
	std::iota(data_p.matrix, data_p.matrix+dim*n, 1);
	std::for_each(data_p.matrix, data_p.matrix+dim*n, [&n](float64_t& val) { val/=n; });

	SGMatrix<float64_t> data_q(dim, n);
	std::iota(data_q.matrix, data_q.matrix+dim*n, n+1);
	std::for_each(data_q.matrix, data_q.matrix+dim*n, [&n](float64_t& val) { val/=2*n; });

	auto feats_p=new CDenseFeatures<float64_t>(data_p);
	auto feats_q=new CDenseFeatures<float64_t>(data_q);
	auto feats=feats_p->create_merged_copy(feats_q);
	SG_REF(feats);
	SG_UNREF(feats_p);
	SG_UNREF(feats_q);

	auto kernel=some<CGaussianKernel>();
	kernel->set_width(2.0);

	kernel->init(feats, feats);
	auto kernel_matrix=kernel->get_kernel_matrix<float32_t>();

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=n;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGVector<float32_t> result_1=permutation_mmd(kernel_matrix);

	auto compute_mmd=ComputeMMD();
	compute_mmd.m_n_x=n;
	compute_mmd.m_n_y=n;
	compute_mmd.m_stype=stype;

	Map<MatrixXf> map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);
	SGVector<float32_t> result_2(num_null_samples);
	sg_rand->set_seed(12345);
	for (auto i=0; i<num_null_samples; ++i)
	{
		PermutationMatrix<Dynamic, Dynamic> perm(kernel_matrix.num_rows);
		perm.setIdentity();
		SGVector<int> perminds(perm.indices().data(), perm.indices().size(), false);
		CMath::permute(perminds);
		MatrixXf permuted = perm.transpose()*map*perm;
		SGMatrix<float32_t> permuted_km(permuted.data(), permuted.rows(), permuted.cols(), false);
		result_2[i]=compute_mmd(permuted_km);
	}

	SGVector<index_t> inds(kernel_matrix.num_rows);
	SGVector<float32_t> result_3(num_null_samples);
	sg_rand->set_seed(12345);
	for (auto i=0; i<num_null_samples; ++i)
	{
		std::iota(inds.vector, inds.vector+inds.vlen, 0);
		CMath::permute(inds);
		feats->add_subset(inds);
		kernel->init(feats, feats);
		kernel_matrix=kernel->get_kernel_matrix<float32_t>();
		result_3[i]=compute_mmd(kernel_matrix);
		feats->remove_subset();
	}

	for (auto i=0; i<num_null_samples; ++i)
	{
		EXPECT_NEAR(result_1[i], result_2[i], 1E-6);
		EXPECT_NEAR(result_1[i], result_3[i], 1E-6);
	}

	SG_UNREF(feats);
}

TEST(PermutationMMD, precomputed_vs_non_precomputed_single_kernel)
{
	const index_t dim=2;
	const index_t n=8;
	const index_t m=8;
	const index_t num_null_samples=5;
	const auto stype=EStatisticType::BIASED_FULL;

	SGMatrix<float64_t> data_p(dim, n);
	std::iota(data_p.matrix, data_p.matrix+dim*n, 1);
	std::for_each(data_p.matrix, data_p.matrix+dim*n, [&n](float64_t& val) { val/=n; });

	SGMatrix<float64_t> data_q(dim, m);
	std::iota(data_q.matrix, data_q.matrix+dim*m, n+1);
	std::for_each(data_q.matrix, data_q.matrix+dim*m, [&m](float64_t& val) { val/=2*m; });

	auto feats_p=new CDenseFeatures<float64_t>(data_p);
	auto feats_q=new CDenseFeatures<float64_t>(data_q);
	auto feats=feats_p->create_merged_copy(feats_q);
	SG_REF(feats);
	SG_UNREF(feats_p);
	SG_UNREF(feats_q);

	auto kernel=some<CGaussianKernel>();
	kernel->set_width(2.0);

	kernel->init(feats, feats);
	auto kernel_matrix=kernel->get_kernel_matrix<float32_t>();

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=m;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGVector<float32_t> result_1=permutation_mmd(kernel_matrix);

	sg_rand->set_seed(12345);
	SGVector<float32_t> result_2=permutation_mmd(Kernel(kernel));

	EXPECT_TRUE(result_1.size()==result_2.size());
	for (auto i=0; i<result_1.size(); ++i)
		EXPECT_NEAR(result_1[i], result_2[i], 1E-6);

	SG_UNREF(feats);
}

TEST(PermutationMMD, biased_full_multi_kernel)
{
	const index_t n=24;
	const index_t m=15;
	const index_t dim=2;
	const index_t num_null_samples=5;
	const index_t num_kernels=4;
	const index_t cache_size=10;
	const float64_t difference=0.5;
	const auto stype=EStatisticType::BIASED_FULL;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(n);
	auto feats_q=gen_q->get_streamed_features(m);
	auto merged_feats=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_p, feats_q));
	SG_REF(merged_feats);

	KernelManager kernel_mgr;
	for (auto i=0; i<num_kernels; ++i)
	{
		auto width=CMath::pow(2, i);
		auto kernel=new CGaussianKernel(cache_size, width);
		kernel_mgr.push_back(kernel);
	}
	auto distance_instance=kernel_mgr.get_distance_instance();
	distance_instance->init(merged_feats, merged_feats);
	auto precomputed_distance=some<CCustomDistance>();
	auto distance_matrix=distance_instance->get_distance_matrix<float32_t>();
	precomputed_distance->set_triangle_distance_matrix_from_full(distance_matrix.data(), n+m, n+m);
	SG_UNREF(distance_instance);
	kernel_mgr.set_precomputed_distance(precomputed_distance);

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=m;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGMatrix<float32_t> null_samples=permutation_mmd(kernel_mgr);
	kernel_mgr.unset_precomputed_distance();

	ASSERT_EQ(null_samples.num_cols, num_kernels);
	ASSERT_EQ(null_samples.num_rows, num_null_samples);

	for (auto k=0; k<num_kernels; ++k)
	{
		CKernel* kernel=kernel_mgr.kernel_at(k);
		kernel->init(merged_feats, merged_feats);
		sg_rand->set_seed(12345);
		SGVector<float32_t> curr_null_samples=permutation_mmd(kernel->get_kernel_matrix<float32_t>());

		ASSERT_EQ(curr_null_samples.size(), null_samples.num_rows);
		for (auto i=0; i<num_null_samples; ++i)
			EXPECT_NEAR(null_samples(i, k), curr_null_samples[i], 1E-5);

		kernel->remove_lhs_and_rhs();
	}
	SG_UNREF(merged_feats);
}

TEST(PermutationMMD, unbiased_full_multi_kernel)
{
	const index_t n=24;
	const index_t m=15;
	const index_t dim=2;
	const index_t num_null_samples=5;
	const index_t num_kernels=4;
	const index_t cache_size=10;
	const float64_t difference=0.5;
	const auto stype=EStatisticType::UNBIASED_FULL;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(n);
	auto feats_q=gen_q->get_streamed_features(m);
	auto merged_feats=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_p, feats_q));
	SG_REF(merged_feats);

	KernelManager kernel_mgr;
	for (auto i=0; i<num_kernels; ++i)
	{
		auto width=CMath::pow(2, i);
		auto kernel=new CGaussianKernel(cache_size, width);
		kernel_mgr.push_back(kernel);
	}
	auto distance_instance=kernel_mgr.get_distance_instance();
	distance_instance->init(merged_feats, merged_feats);
	auto precomputed_distance=some<CCustomDistance>();
	auto distance_matrix=distance_instance->get_distance_matrix<float32_t>();
	precomputed_distance->set_triangle_distance_matrix_from_full(distance_matrix.data(), n+m, n+m);
	SG_UNREF(distance_instance);
	kernel_mgr.set_precomputed_distance(precomputed_distance);

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=m;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGMatrix<float32_t> null_samples=permutation_mmd(kernel_mgr);
	kernel_mgr.unset_precomputed_distance();

	ASSERT_EQ(null_samples.num_cols, num_kernels);
	ASSERT_EQ(null_samples.num_rows, num_null_samples);

	for (auto k=0; k<num_kernels; ++k)
	{
		CKernel* kernel=kernel_mgr.kernel_at(k);
		kernel->init(merged_feats, merged_feats);
		sg_rand->set_seed(12345);
		SGVector<float32_t> curr_null_samples=permutation_mmd(kernel->get_kernel_matrix<float32_t>());

		ASSERT_EQ(curr_null_samples.size(), null_samples.num_rows);
		for (auto i=0; i<num_null_samples; ++i)
			EXPECT_NEAR(null_samples(i, k), curr_null_samples[i], 1E-5);

		kernel->remove_lhs_and_rhs();
	}
	SG_UNREF(merged_feats);
}

TEST(PermutationMMD, unbiased_incomplete_multi_kernel)
{
	const index_t n=18;
	const index_t m=18;
	const index_t dim=2;
	const index_t num_null_samples=5;
	const index_t num_kernels=4;
	const index_t cache_size=10;
	const float64_t difference=0.5;
	const auto stype=EStatisticType::UNBIASED_INCOMPLETE;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(n);
	auto feats_q=gen_q->get_streamed_features(m);
	auto merged_feats=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_p, feats_q));
	SG_REF(merged_feats);

	KernelManager kernel_mgr;
	for (auto i=0; i<num_kernels; ++i)
	{
		auto width=CMath::pow(2, i);
		auto kernel=new CGaussianKernel(cache_size, width);
		kernel_mgr.push_back(kernel);
	}
	auto distance_instance=kernel_mgr.get_distance_instance();
	distance_instance->init(merged_feats, merged_feats);
	auto precomputed_distance=some<CCustomDistance>();
	auto distance_matrix=distance_instance->get_distance_matrix<float32_t>();
	precomputed_distance->set_triangle_distance_matrix_from_full(distance_matrix.data(), n+m, n+m);
	SG_UNREF(distance_instance);
	kernel_mgr.set_precomputed_distance(precomputed_distance);

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_n_x=n;
	permutation_mmd.m_n_y=m;
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	SGMatrix<float32_t> null_samples=permutation_mmd(kernel_mgr);
	kernel_mgr.unset_precomputed_distance();

	ASSERT_EQ(null_samples.num_cols, num_kernels);
	ASSERT_EQ(null_samples.num_rows, num_null_samples);

	for (auto k=0; k<num_kernels; ++k)
	{
		CKernel* kernel=kernel_mgr.kernel_at(k);
		kernel->init(merged_feats, merged_feats);
		sg_rand->set_seed(12345);
		SGVector<float32_t> curr_null_samples=permutation_mmd(kernel->get_kernel_matrix<float32_t>());

		ASSERT_EQ(curr_null_samples.size(), null_samples.num_rows);
		for (auto i=0; i<num_null_samples; ++i)
			EXPECT_NEAR(null_samples(i, k), curr_null_samples[i], 1E-5);

		kernel->remove_lhs_and_rhs();
	}
	SG_UNREF(merged_feats);
}
