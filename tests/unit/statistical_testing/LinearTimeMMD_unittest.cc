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
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LinearTimeMMD, biased_same_num_samples)
{
	const index_t m=4;
	const index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	// normalise data
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	// assert matlab result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.090438791828373444, 1E-5);
}

TEST(LinearTimeMMD, unbiased_same_num_samples)
{
	const index_t m=4;
	const index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	// normalise data
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_FULL);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	// assert matlab result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.066491458266665582, 1E-5);
}

TEST(LinearTimeMMD, incomplete_same_num_samples)
{
	const index_t m=4;
	const index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	// normalise data
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_INCOMPLETE);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	// assert local machine computed result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.083423196012644057, 1E-5);
}

TEST(LinearTimeMMD, biased_different_null_samples)
{
	const index_t m=4;
	const index_t n=6;
	const index_t d=3;
	SGMatrix<float64_t> data(d,m+n);
	for (index_t i=0; i<d*(m+n); ++i)
		data.matrix[i]=i;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*n);

	// normalise data
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
		data_p.matrix[i]/=max_p;
	for (index_t i=0; i<d*n; ++i)
		data_q.matrix[i]/=max_q;

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	// assert matlab result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.06525051478776954, 1E-5);
}

TEST(LinearTimeMMD, unbiased_different_null_samples)
{
	const index_t m=4;
	const index_t n=6;
	const index_t d=3;
	SGMatrix<float64_t> data(d,m+n);
	for (index_t i=0; i<d*(m+n); ++i)
		data.matrix[i]=i;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*n);

	// normalise data
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
		data_p.matrix[i]/=max_p;
	for (index_t i=0; i<d*n; ++i)
		data_q.matrix[i]/=max_q;

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_FULL);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	// assert matlab result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.039823645725702045, 1E-5);
}

TEST(LinearTimeMMD, compute_variance_null)
{
	const index_t m=8;
	const index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	// normalise data
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(features_p, features_q);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	// assert local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_FULL);
	float64_t var=mmd->compute_variance();
	EXPECT_NEAR(var, 0.0022330284118652344, 1E-10);

	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	var=mmd->compute_variance();
	EXPECT_NEAR(var, 0.0022330284118652344, 1E-10);

	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_INCOMPLETE);
	var=mmd->compute_variance();
	EXPECT_NEAR(var, 0.0022330284118652344, 1E-10);
}

TEST(LinearTimeMMD, perform_test_permutation_biased_full)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_PERMUTATION);

	// compute p-value using permutation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	float64_t p_value=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value, 0.0, 1E-10);
}

TEST(LinearTimeMMD, perform_test_permutation_unbiased_full)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_PERMUTATION);

	// compute p-value using permutation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_FULL);
	float64_t p_value=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value, 0.0, 1E-10);
}

TEST(LinearTimeMMD, perform_test_permutation_unbiased_incomplete)
{
	const index_t m=20;
	const index_t n=20;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_PERMUTATION);

	// compute p-value using permutation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_INCOMPLETE);
	float64_t p_value=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value, 0.59999999999999998, 1E-10);
}

TEST(LinearTimeMMD, perform_test_gaussian_biased_full)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_MMD1_GAUSSIAN);

	// biased case

	// compute p-value using Gaussian approximation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_BIASED_FULL);
	float64_t p_value_gaussian=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value_gaussian, 0.0, 1E-10);
}

TEST(LinearTimeMMD, perform_test_gaussian_unbiased_full)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_MMD1_GAUSSIAN);

	// unbiased case

	// compute p-value using spectrum approximation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_FULL);
	float64_t p_value_gaussian=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value_gaussian, 0.060947882185221292, 1E-10);
}

TEST(LinearTimeMMD, perform_test_gaussian_unbiased_incomplete)
{
	const index_t m=20;
	const index_t n=20;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CLinearTimeMMD>(gen_p, gen_q);
	mmd->set_num_samples_p(m);
	mmd->set_num_samples_q(n);
	mmd->set_num_blocks_per_burst(1000);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::NAM_MMD1_GAUSSIAN);

	// unbiased case

	// compute p-value using spectrum approximation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::ST_UNBIASED_INCOMPLETE);
	float64_t p_value_gaussian=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value_gaussian, 0.40645354706402292, 1E-10);
}
