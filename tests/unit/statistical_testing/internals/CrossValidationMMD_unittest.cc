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
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/CrossValidationMMD.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

TEST(CrossValidationMMD, biased_full)
{
	const index_t n=24;
	const index_t m=15;
	const index_t dim=2;
	const index_t num_null_samples=5;
	const index_t num_folds=3;
	const index_t num_runs=2;
	const index_t num_kernels=4;
	const index_t cache_size=10;
	const float64_t difference=0.5;
	const float64_t alpha=0.05;
	const auto stype=EStatisticType::ST_BIASED_FULL;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(n);
	auto feats_q=gen_q->get_streamed_features(m);
	auto merged_feats=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_p, feats_q));

	KernelManager kernel_mgr;
	for (auto i=0; i<num_kernels; ++i)
	{
		auto width=pow(2, i);
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
	auto cv=CrossValidationMMD(n, m, num_folds, num_null_samples);
	cv.m_stype=stype;
	cv.m_alpha=alpha;
	cv.m_num_runs=num_runs;
	cv.m_rejections=SGMatrix<float64_t>(num_runs*num_folds, num_kernels);
	sg_rand->set_seed(12345);
	cv(kernel_mgr);
	kernel_mgr.unset_precomputed_distance();

	SGVector<int64_t> dummy_labels_p(n);
	SGVector<int64_t> dummy_labels_q(m);

	auto kfold_p=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_p), num_folds);
	auto kfold_q=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_q), num_folds);

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	for (auto k=0; k<num_kernels; ++k)
	{
		CKernel* kernel=kernel_mgr.kernel_at(k);
		for (auto current_run=0; current_run<num_runs; ++current_run)
		{
			kfold_p->build_subsets();
			kfold_q->build_subsets();

			for (auto current_fold=0; current_fold<num_folds; ++current_fold)
			{
				auto current_train_subset_p=kfold_p->generate_subset_inverse(current_fold);
				auto current_train_subset_q=kfold_q->generate_subset_inverse(current_fold);

				feats_p->add_subset(current_train_subset_p);
				feats_q->add_subset(current_train_subset_q);

				permutation_mmd.m_n_x=feats_p->get_num_vectors();
				permutation_mmd.m_n_y=feats_q->get_num_vectors();

				auto current_merged_feats=static_cast<CDenseFeatures<float64_t>*>
					(FeaturesUtil::create_merged_copy(feats_p, feats_q));

				kernel->init(current_merged_feats, current_merged_feats);
				auto p_value=permutation_mmd.p_value(kernel->get_kernel_matrix<float32_t>());

				EXPECT_EQ(cv.m_rejections(current_run*num_folds+current_fold, k), p_value<alpha);

				kernel->remove_lhs_and_rhs();
				feats_p->remove_subset();
				feats_q->remove_subset();
			}
		}
	}
}

TEST(CrossValidationMMD, unbiased_full)
{
	const index_t n=24;
	const index_t m=15;
	const index_t dim=2;
	const index_t num_null_samples=5;
	const index_t num_folds=3;
	const index_t num_runs=2;
	const index_t num_kernels=4;
	const index_t cache_size=10;
	const float64_t difference=0.5;
	const float64_t alpha=0.05;
	const auto stype=EStatisticType::ST_UNBIASED_FULL;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(n);
	auto feats_q=gen_q->get_streamed_features(m);
	auto merged_feats=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_p, feats_q));

	KernelManager kernel_mgr;
	for (auto i=0; i<num_kernels; ++i)
	{
		auto width=pow(2, i);
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
	auto cv=CrossValidationMMD(n, m, num_folds, num_null_samples);
	cv.m_stype=stype;
	cv.m_alpha=alpha;
	cv.m_num_runs=num_runs;
	cv.m_rejections=SGMatrix<float64_t>(num_runs*num_folds, num_kernels);
	sg_rand->set_seed(12345);
	cv(kernel_mgr);
	kernel_mgr.unset_precomputed_distance();

	SGVector<int64_t> dummy_labels_p(n);
	SGVector<int64_t> dummy_labels_q(m);

	auto kfold_p=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_p), num_folds);
	auto kfold_q=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_q), num_folds);

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	for (auto k=0; k<num_kernels; ++k)
	{
		CKernel* kernel=kernel_mgr.kernel_at(k);
		for (auto current_run=0; current_run<num_runs; ++current_run)
		{
			kfold_p->build_subsets();
			kfold_q->build_subsets();

			for (auto current_fold=0; current_fold<num_folds; ++current_fold)
			{
				auto current_train_subset_p=kfold_p->generate_subset_inverse(current_fold);
				auto current_train_subset_q=kfold_q->generate_subset_inverse(current_fold);

				feats_p->add_subset(current_train_subset_p);
				feats_q->add_subset(current_train_subset_q);

				permutation_mmd.m_n_x=feats_p->get_num_vectors();
				permutation_mmd.m_n_y=feats_q->get_num_vectors();

				auto current_merged_feats=static_cast<CDenseFeatures<float64_t>*>
					(FeaturesUtil::create_merged_copy(feats_p, feats_q));

				kernel->init(current_merged_feats, current_merged_feats);
				auto p_value=permutation_mmd.p_value(kernel->get_kernel_matrix<float32_t>());

				EXPECT_EQ(cv.m_rejections(current_run*num_folds+current_fold, k), p_value<alpha);

				kernel->remove_lhs_and_rhs();
				feats_p->remove_subset();
				feats_q->remove_subset();
			}
		}
	}
}

TEST(CrossValidationMMD, unbiased_incomplete)
{
	const index_t n=18;
	const index_t m=18;
	const index_t dim=2;
	const index_t num_null_samples=5;
	const index_t num_folds=3;
	const index_t num_runs=2;
	const index_t num_kernels=4;
	const index_t cache_size=10;
	const float64_t difference=0.5;
	const float64_t alpha=0.05;
	const auto stype=EStatisticType::ST_UNBIASED_INCOMPLETE;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(n);
	auto feats_q=gen_q->get_streamed_features(m);
	auto merged_feats=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_p, feats_q));

	KernelManager kernel_mgr;
	for (auto i=0; i<num_kernels; ++i)
	{
		auto width=pow(2, i);
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
	auto cv=CrossValidationMMD(n, m, num_folds, num_null_samples);
	cv.m_stype=stype;
	cv.m_alpha=alpha;
	cv.m_num_runs=num_runs;
	cv.m_rejections=SGMatrix<float64_t>(num_runs*num_folds, num_kernels);
	sg_rand->set_seed(12345);
	cv(kernel_mgr);
	kernel_mgr.unset_precomputed_distance();

	SGVector<int64_t> dummy_labels_p(n);
	SGVector<int64_t> dummy_labels_q(m);

	auto kfold_p=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_p), num_folds);
	auto kfold_q=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_q), num_folds);

	auto permutation_mmd=PermutationMMD();
	permutation_mmd.m_stype=stype;
	permutation_mmd.m_num_null_samples=num_null_samples;

	sg_rand->set_seed(12345);
	for (auto k=0; k<num_kernels; ++k)
	{
		CKernel* kernel=kernel_mgr.kernel_at(k);
		for (auto current_run=0; current_run<num_runs; ++current_run)
		{
			kfold_p->build_subsets();
			kfold_q->build_subsets();

			for (auto current_fold=0; current_fold<num_folds; ++current_fold)
			{
				auto current_train_subset_p=kfold_p->generate_subset_inverse(current_fold);
				auto current_train_subset_q=kfold_q->generate_subset_inverse(current_fold);

				feats_p->add_subset(current_train_subset_p);
				feats_q->add_subset(current_train_subset_q);

				permutation_mmd.m_n_x=feats_p->get_num_vectors();
				permutation_mmd.m_n_y=feats_q->get_num_vectors();

				auto current_merged_feats=static_cast<CDenseFeatures<float64_t>*>
					(FeaturesUtil::create_merged_copy(feats_p, feats_q));

				kernel->init(current_merged_feats, current_merged_feats);
				auto p_value=permutation_mmd.p_value(kernel->get_kernel_matrix<float32_t>());

				EXPECT_EQ(cv.m_rejections(current_run*num_folds+current_fold, k), p_value<alpha);

				kernel->remove_lhs_and_rhs();
				feats_p->remove_subset();
				feats_q->remove_subset();
			}
		}
	}
}
