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
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(QuadraticTimeMMD, biased)
{
	index_t m=8;
	index_t d=3;
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
	auto mmd=some<CQuadraticTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::BIASED_FULL);
	mmd->set_kernel(kernel);

	// assert matlab result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.17882546486779649, 1E-5);
}

TEST(QuadraticTimeMMD, unbiased)
{
	index_t m=8;
	index_t d=3;
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
	auto mmd=some<CQuadraticTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::UNBIASED_FULL);
	mmd->set_kernel(kernel);

	// assert matlab result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.13440094336133723, 1E-5);
}

TEST(QuadraticTimeMMD, incomplete)
{
	index_t m=8;
	index_t d=3;
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
	auto mmd=some<CQuadraticTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::UNBIASED_INCOMPLETE);
	mmd->set_kernel(kernel);

	// assert local machine computed result
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.16743977201175841, 1E-5);
}

TEST(QuadraticTimeMMD, unbiased_different_num_samples)
{
	const index_t m=5;
	const index_t n=6;
	const index_t d=1;
	float64_t data[] = {0.61318059, -0.69222999, 0.94424411, -0.48769626,
		-0.00709551,  0.35025598, 0.20741384, -0.63622519, -1.21315264,
	   	-0.77349617, -0.42707091};

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data[0]), sizeof(float64_t)*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data[m]), sizeof(float64_t)*n);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);

	// create MMD instance, convienience constructor
	auto mmd=some<CQuadraticTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::UNBIASED_FULL);
	mmd->set_kernel(kernel);

	// assert python result at
	// https://github.com/lambday/shogun-hypothesis-testing/blob/master/mmd.py
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, -0.037500338130199401, 1E-5);
}

TEST(QuadraticTimeMMD, biased_different_num_samples)
{
	const index_t m=5;
	const index_t n=6;
	const index_t d=1;
	float64_t data[] = {-0.47616889, -2.1767364, -0.04185537, -1.20787529,
		1.94875193, -0.16695709, 2.51282666, -0.58116389, 1.52366887,
		0.18985099, 0.76120258};

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data[0]), sizeof(float64_t)*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data[m]), sizeof(float64_t)*n);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// shoguns kernel width is different
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);

	// create MMD instance, convienience constructor
	auto mmd=some<CQuadraticTimeMMD>(features_p, features_q);
	mmd->set_statistic_type(EStatisticType::BIASED_FULL);
	mmd->set_kernel(kernel);

	// assert python result at
	// https://github.com/lambday/shogun-hypothesis-testing/blob/master/mmd.py
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.54418915736201567, 1E-5);
}

TEST(QuadraticTimeMMD, compute_variance_null)
{
	index_t m=8;
	index_t d=3;
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
	auto mmd=some<CQuadraticTimeMMD>(features_p, features_q);
	mmd->set_kernel(kernel);

	// assert local machine computed result
	mmd->set_statistic_type(EStatisticType::UNBIASED_FULL);
	float64_t var=mmd->compute_variance();
	EXPECT_NEAR(var, 0.0064888052500351456, 1E-10);

	mmd->set_statistic_type(EStatisticType::BIASED_FULL);
	var=mmd->compute_variance();
	EXPECT_NEAR(var, 0.0071464012090942663, 1E-10);

	mmd->set_statistic_type(EStatisticType::UNBIASED_INCOMPLETE);
	var=mmd->compute_variance();
	EXPECT_NEAR(var, 0.0064888052500342575, 1E-10);
}

TEST(QuadraticTimeMMD, perform_test_permutation_biased_full)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	// stream some data from generator
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(n);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CQuadraticTimeMMD>(feat_p, feat_q);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::PERMUTATION);

	// compute p-value using permutation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::BIASED_FULL);
	float64_t p_value=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value, 0.0, 1E-10);
}

TEST(QuadraticTimeMMD, perform_test_permutation_unbiased_full)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	// stream some data from generator
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(n);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CQuadraticTimeMMD>(feat_p, feat_q);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::PERMUTATION);

	// compute p-value using permutation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::UNBIASED_FULL);
	float64_t p_value=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value, 0.0, 1E-10);
}

TEST(QuadraticTimeMMD, perform_test_permutation_unbiased_incomplete)
{
	const index_t m=20;
	const index_t n=20;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	// stream some data from generator
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(n);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CQuadraticTimeMMD>(feat_p, feat_q);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::PERMUTATION);

	// compute p-value using permutation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::UNBIASED_INCOMPLETE);
	float64_t p_value=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value, 0.0, 1E-10);
}

TEST(QuadraticTimeMMD, perform_test_spectrum)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	// use fixed seed
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	// streaming data generator for mean shift distributions
	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	// stream some data from generator
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(n);

	// shoguns kernel width is different
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance, convienience constructor
	auto mmd=some<CQuadraticTimeMMD>(feat_p, feat_q);
	mmd->set_kernel(kernel);

	index_t num_null_samples=10;
	index_t num_eigenvalues=10;
	mmd->set_num_null_samples(num_null_samples);
	mmd->set_null_approximation_method(ENullApproximationMethod::MMD2_SPECTRUM);
	mmd->spectrum_set_num_eigenvalues(num_eigenvalues);

	// biased case

	// compute p-value using spectrum approximation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::BIASED_FULL);
	float64_t p_value_spectrum=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value_spectrum, 0.0, 1E-10);

	// unbiased case

	// compute p-value using spectrum approximation for null distribution and
	// assert against local machine computed result
	mmd->set_statistic_type(EStatisticType::UNBIASED_FULL);
	p_value_spectrum=mmd->compute_p_value(mmd->compute_statistic());
	EXPECT_NEAR(p_value_spectrum, 0.0, 1E-10);
}
