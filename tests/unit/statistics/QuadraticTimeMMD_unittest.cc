/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/eigen3.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(QuadraticTimeMMD,test_quadratic_mmd_biased)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* shoguns kernel width is different */
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features_p, features_q);
	mmd->set_statistic_type(BIASED);

	/* assert matlab result */
	float64_t statistic=mmd->compute_statistic();
	//SG_SPRINT("statistic=%f\n", statistic);
	EXPECT_NEAR(statistic, 0.357650929735592, 10E-15);

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(features_p);
	SG_UNREF(features_q);
}

TEST(QuadraticTimeMMD,test_quadratic_mmd_unbiased)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* shoguns kernel width is different */
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features_p, features_q);
	mmd->set_statistic_type(UNBIASED);

	/* assert matlab result */
	float64_t statistic=mmd->compute_statistic();
	//SG_SPRINT("statistic=%f\n", statistic);
	float64_t difference=statistic-0.268801886722675;
	EXPECT_LE(CMath::abs(difference), 10E-15);

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(features_p);
	SG_UNREF(features_q);
}

TEST(QuadraticTimeMMD, test_quadratic_mmd_unbiased_different_num_samples)
{
	const index_t m=5;
	const index_t n=6;
	const index_t d=1;
	float64_t data[] = {0.61318059, -0.69222999, 0.94424411, -0.48769626,
		-0.00709551,  0.35025598, 0.20741384, -0.63622519, -1.21315264,
	   	-0.77349617, -0.42707091};

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data[0]), sizeof(float64_t)*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data[m]), sizeof(float64_t)*n);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features_p, features_q);
	mmd->set_statistic_type(UNBIASED);

	/* assert python result at
	 * https://github.com/lambday/shogun-hypothesis-testing/blob/master/mmd.py */
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, -0.151251364436, 1E-9);

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(features_p);
	SG_UNREF(features_q);
}

TEST(QuadraticTimeMMD, test_quadratic_mmd_biased_different_num_samples)
{
	const index_t m=5;
	const index_t n=6;
	const index_t d=1;
	float64_t data[] = {-0.47616889, -2.1767364, -0.04185537, -1.20787529,
		1.94875193, -0.16695709, 2.51282666, -0.58116389, 1.52366887,
		0.18985099, 0.76120258};

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data[0]), sizeof(float64_t)*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data[m]), sizeof(float64_t)*n);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features_p, features_q);
	mmd->set_statistic_type(BIASED);

	/* assert python result at
	 * https://github.com/lambday/shogun-hypothesis-testing/blob/master/mmd.py */
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 2.1948962593, 1E-8);

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(features_p);
	SG_UNREF(features_q);
}

#ifdef HAVE_EIGEN3
TEST(QuadraticTimeMMD, null_approximation_spectrum_different_num_samples)
{
	const index_t m=20;
	const index_t n=30;
	const index_t dim=3;

	/* use fixed seed */
	sg_rand->set_seed(12345);

	float64_t difference=0.5;

	/* streaming data generator for mean shift distributions */
	CMeanShiftDataGenerator* gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	CMeanShiftDataGenerator* gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	/* stream some data from generator */
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(n);

	/* shoguns kernel width is different */
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, feat_p, feat_q);

	index_t num_null_samples=250;
	index_t num_eigenvalues=10;
	mmd->set_num_samples_spectrum(num_null_samples);
	mmd->set_null_approximation_method(MMD2_SPECTRUM);
	mmd->set_num_eigenvalues_spectrum(num_eigenvalues);

	/* biased case */

	/* compute p-value using spectrum approximation for null distribution and
	 * assert against local machine computed result */
	mmd->set_statistic_type(BIASED);
	float64_t p_value_spectrum=mmd->perform_test();
	EXPECT_NEAR(p_value_spectrum, 0.0, 1E-10);

	/* unbiased case */

	/* compute p-value using spectrum approximation for null distribution and
	 * assert against local machine computed result */
	mmd->set_statistic_type(UNBIASED);
	p_value_spectrum=mmd->perform_test();
	EXPECT_NEAR(p_value_spectrum, 0.004, 1E-10);

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(feat_p);
	SG_UNREF(feat_q);
	SG_UNREF(gen_p);
	SG_UNREF(gen_q);
}
#endif // HAVE_EIGEN3

TEST(QuadraticTimeMMD,test_quadratic_mmd_precomputed_kernel)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);
	CFeatures* p_and_q=features_p->create_merged_copy(features_q);
	SG_REF(p_and_q);

	/* shoguns kernel width is different */
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, p_and_q, m);
	mmd->set_num_null_samples(10);

	/* use fixed seed */
	sg_rand->set_seed(12345);
	SGVector<float64_t> null_samples=mmd->sample_null();

	float64_t mean=CStatistics::mean(null_samples);
	float64_t var=CStatistics::variance(null_samples);

	//SG_SPRINT("mean %f, var %f\n", mean, var);

	/* now again but with a precomputed kernel, same features.
	 * This avoids re-computing the kernel matrix in every permutation
	 * iteration and should be num_iterations times faster */

	/* re-init kernel before kernel matrix is computed: this is due to a design
	 * error in subsets and should be worked on! */
	kernel->init(p_and_q, p_and_q);
	CCustomKernel* precomputed_kernel=new CCustomKernel(kernel);
	SG_UNREF(mmd);
	mmd=new CQuadraticTimeMMD(precomputed_kernel, p_and_q, m);
	mmd->set_statistic_type(UNBIASED);
	mmd->set_num_null_samples(10);
	sg_rand->set_seed(12345);
	null_samples=mmd->sample_null();

	/* assert that results do not change */
	//SG_SPRINT("mean %f, var %f\n", CStatistics::mean(null_samples),
	//		CStatistics::variance(null_samples));
	EXPECT_LE(CMath::abs(mean-CStatistics::mean(null_samples)), 10E-8);
	EXPECT_LE(CMath::abs(var-CStatistics::variance(null_samples)), 10E-8);

	SG_UNREF(mmd);
	SG_UNREF(features_p);
	SG_UNREF(features_q);
	SG_UNREF(p_and_q);
}

#ifdef HAVE_EIGEN3
TEST(QuadraticTimeMMD,custom_kernel_vs_normal_kernel)
{
	/* number of examples kept low in order to make things fast */
	index_t m=20;
	index_t dim=2;
	float64_t difference=0.5;

	/* streaming data generator for mean shift distributions */
	CMeanShiftDataGenerator* gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	CMeanShiftDataGenerator* gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	/* stream some data from generator */
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(m);

	/* set kernel a-priori. usually one would do some kernel selection. See
	 * other examples for this. */
	float64_t width=10;
	CGaussianKernel* kernel=new CGaussianKernel(10, width);

	/* create quadratic time mmd instance. Note that this constructor
	 * copies p and q and does not reference them */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, feat_p, feat_q);

	/* set up for a precomputed custom kernel using merged features p_and_q */
	CGaussianKernel* kernel2=new CGaussianKernel(10, width);
	CFeatures* p_and_q=mmd->get_p_and_q();
	kernel2->init(p_and_q, p_and_q);
	CCustomKernel* precomputed=new CCustomKernel(kernel2);
	CQuadraticTimeMMD* mmd2=new CQuadraticTimeMMD(precomputed, m);
	SG_UNREF(p_and_q);
	SG_UNREF(kernel2);

	/* perform test: compute p-value and test if null-hypothesis is rejected for
	 * a test level of 0.05 */
	float64_t alpha=0.05;

	mmd->set_null_approximation_method(PERMUTATION);
	mmd->set_statistic_type(BIASED);
	mmd->set_num_null_samples(3);
	mmd->set_num_eigenvalues_spectrum(3);
	mmd->set_num_samples_spectrum(250);

	mmd2->set_null_approximation_method(PERMUTATION);
	mmd2->set_statistic_type(BIASED);
	mmd2->set_num_null_samples(3);
	mmd2->set_num_eigenvalues_spectrum(3);
	mmd2->set_num_samples_spectrum(250);

	/* compute tpye I and II error using normal and precomputed kernel */
	index_t num_trials=3;

	SGVector<index_t> inds(2*m);
	inds.range_fill();

	/* use fixed seed */
	CMath::init_random(1);
	for (index_t i=0; i<num_trials; ++i)
	{
		/* this effectively means that p=q - rejecting is tpye I error */
		inds.permute();

		/* setting seed for Gaussian samples used in spectrum approximation method */
		sg_rand->set_seed(1);

		/* first, we compute using normal kernel */
		p_and_q->add_subset(inds);
		float64_t type_I_mmds=mmd->compute_statistic();
		mmd->set_null_approximation_method(PERMUTATION);
		float64_t type_I_threshs_boot=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_SPECTRUM);
		float64_t type_I_threshs_spectrum=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_GAMMA);
		float64_t type_I_threshs_gamma=mmd->compute_threshold(alpha);
		p_and_q->remove_subset();

		float64_t type_II_mmds=mmd->compute_statistic();
		mmd->set_null_approximation_method(PERMUTATION);
		float64_t type_II_threshs_boot=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_SPECTRUM);
		float64_t type_II_threshs_spectrum=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_GAMMA);
		float64_t type_II_threshs_gamma=mmd->compute_threshold(alpha);

		/* now compute using precomputed custom kernel */

		/* setting seed for Gaussian samples used in spectrum approximation method */
		sg_rand->set_seed(1);

		precomputed->add_row_subset(inds);
		precomputed->add_col_subset(inds);
		float64_t type_I_mmds_pre=mmd2->compute_statistic();
		mmd2->set_null_approximation_method(PERMUTATION);
		float64_t type_I_threshs_boot_pre=mmd2->compute_threshold(alpha);
		mmd2->set_null_approximation_method(MMD2_SPECTRUM);
		float64_t type_I_threshs_spectrum_pre=mmd2->compute_threshold(alpha);
		mmd2->set_null_approximation_method(MMD2_GAMMA);
		float64_t type_I_threshs_gamma_pre=mmd2->compute_threshold(alpha);
		precomputed->remove_row_subset();
		precomputed->remove_col_subset();

		float64_t type_II_mmds_pre=mmd2->compute_statistic();
		mmd2->set_null_approximation_method(PERMUTATION);
		float64_t type_II_threshs_boot_pre=mmd2->compute_threshold(alpha);
		mmd2->set_null_approximation_method(MMD2_SPECTRUM);
		float64_t type_II_threshs_spectrum_pre=mmd2->compute_threshold(alpha);
		mmd2->set_null_approximation_method(MMD2_GAMMA);
		float64_t type_II_threshs_gamma_pre=mmd2->compute_threshold(alpha);

		/* assert results from both */
		EXPECT_NEAR(type_I_mmds, type_I_mmds_pre, 1E-6);
		EXPECT_NEAR(type_I_threshs_boot, type_I_threshs_boot_pre, 1E-6);
		EXPECT_NEAR(type_I_threshs_spectrum, type_I_threshs_spectrum_pre, 1E-6);
		EXPECT_NEAR(type_I_threshs_gamma, type_I_threshs_gamma_pre, 1E-6);
		EXPECT_NEAR(type_II_mmds, type_II_mmds_pre, 1E-5);
		EXPECT_NEAR(type_II_threshs_boot, type_II_threshs_boot_pre, 1E-6);
		EXPECT_NEAR(type_II_threshs_spectrum, type_II_threshs_spectrum_pre, 1E-6);
		EXPECT_NEAR(type_II_threshs_gamma, type_II_threshs_gamma_pre, 1E-6);
	}

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(mmd2);
	SG_UNREF(gen_p);
	SG_UNREF(gen_q);

	/* convienience constructor of MMD was used, these were not referenced */
	SG_UNREF(feat_p);
	SG_UNREF(feat_q);
}
#endif // HAVE_EIGEN3
