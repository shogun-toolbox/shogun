/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2012-2013 Heiko Strathmann
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
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KernelSelectionMaxMMD, linear_time_single_kernel_streaming)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_MMD);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 0.03125, 1E-10);
}

TEST(KernelSelectionMaxMMD, quadratic_time_single_kernel_dense)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=gen_p->get_streamed_features(m);
	auto feats_q=gen_q->get_streamed_features(n);

	auto mmd=some<CQuadraticTimeMMD>(feats_p, feats_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_MMD);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 0.25, 1E-10);
}

TEST(KernelSelectionMaxMMD, linear_time_weighted_kernel_streaming)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_MMD, true);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto weighted_kernel=dynamic_cast<CCombinedKernel*>(mmd->get_kernel());
	ASSERT_TRUE(weighted_kernel!=nullptr);
	ASSERT_TRUE(weighted_kernel->get_num_subkernels()==num_kernels);

	SGVector<float64_t> weights=weighted_kernel->get_subkernel_weights();
	for (auto i=0; i<weights.size(); ++i)
		EXPECT_NEAR(weights[i], 0.1, 1E-10);
}

TEST(KernelSelectionMaxTestPower, linear_time_single_kernel_streaming)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_POWER);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 0.03125, 1E-10);
}

TEST(KernelSelectionMaxTestPower, linear_time_weighted_kernel_streaming)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_POWER, true);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto weighted_kernel=dynamic_cast<CCombinedKernel*>(mmd->get_kernel());
	ASSERT_TRUE(weighted_kernel!=nullptr);
	ASSERT_TRUE(weighted_kernel->get_num_subkernels()==num_kernels);

	SGVector<float64_t> weights=weighted_kernel->get_subkernel_weights();
	for (auto i=0; i<weights.size(); ++i)
		EXPECT_NEAR(weights[i], 0.1, 1E-10);
}

TEST(KernelSelectionMaxCrossValidation, quadratic_time_single_kernel_dense)
{
	const index_t m=20;
	const index_t n=20;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=5;
	const index_t num_runs=1;
	const index_t num_folds=3;
	const float64_t train_test_ratio=3;
	const float64_t alpha=0.05;

	sg_rand->set_seed(12345);

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);
	auto feats_p=gen_p->get_streamed_features(m);
	auto feats_q=gen_q->get_streamed_features(n);

	auto mmd=some<CQuadraticTimeMMD>(feats_p, feats_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_PERMUTATION);
	mmd->set_num_null_samples(10);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_CROSS_VALIDATION, num_runs, num_folds, alpha);

	mmd->set_train_test_mode(true);
	mmd->set_train_test_ratio(train_test_ratio);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 0.03125, 1E-10);
}

TEST(KernelSelectionMaxCrossValidation, linear_time_single_kernel_dense)
{
	const index_t m=8;
	const index_t n=12;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;
	const index_t num_runs=1;
	const index_t num_folds=3;
	const float64_t train_test_ratio=3;
	const float64_t alpha=0.05;

	sg_rand->set_seed(12345);

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);
	auto feats_p=gen_p->get_streamed_features(m);
	auto feats_q=gen_q->get_streamed_features(n);

	auto mmd=some<CLinearTimeMMD>(feats_p, feats_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MAXIMIZE_CROSS_VALIDATION, num_runs, num_folds, alpha);

	mmd->set_train_test_mode(true);
	mmd->set_train_test_ratio(train_test_ratio);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 0.03125, 1E-10);
}

TEST(KernelSelectionMedianHeuristic, quadratic_time_single_kernel_dense)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto mmd=some<CQuadraticTimeMMD>(gen_p, gen_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MEDIAN_HEURISTIC);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 1.0, 1E-10);
}

TEST(KernelSelectionMedianHeuristic, linear_time_single_kernel_dense)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;
	const index_t num_kernels=10;

	sg_rand->set_seed(12345);

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	for (auto i=0, sigma=-5; i<num_kernels; ++i, sigma+=1)
	{
		float64_t tau=pow(2, sigma);
		mmd->add_kernel(new CGaussianKernel(10, tau));
	}
	mmd->set_kernel_selection_strategy(EKernelSelectionMethod::KSM_MEDIAN_HEURISTIC);

	mmd->set_train_test_mode(true);
	mmd->select_kernel();
	mmd->set_train_test_mode(false);

	auto selected_kernel=static_cast<CGaussianKernel*>(mmd->get_kernel());
	EXPECT_NEAR(selected_kernel->get_width(), 1.0, 1E-10);
}
