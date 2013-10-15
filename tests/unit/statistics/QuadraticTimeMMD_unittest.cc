/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/Statistics.h>
#include <gtest/gtest.h>

using namespace shogun;

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
	mmd->set_bootstrap_iterations(10);

	/* use fixed seed */
	sg_rand->set_seed(12345);
	SGVector<float64_t> null_samples=mmd->bootstrap_null();

	float64_t mean=CStatistics::mean(null_samples);
	float64_t var=CStatistics::variance(null_samples);

	//SG_SPRINT("mean %f, var %f\n", mean, var);

	/* now again but with a precomputed kernel, same features.
	 * This avoids re-computing the kernel matrix in every bootstrapping
	 * iteration and should be num_iterations times faster */

	/* re-init kernel before kernel matrix is computed: this is due to a design
	 * error in subsets and should be worked on! */
	kernel->init(p_and_q, p_and_q);
	CCustomKernel* precomputed_kernel=new CCustomKernel(kernel);
	SG_UNREF(mmd);
	mmd=new CQuadraticTimeMMD(precomputed_kernel, p_and_q, m);
	mmd->set_statistic_type(UNBIASED);
	mmd->set_bootstrap_iterations(10);
	sg_rand->set_seed(12345);
	null_samples=mmd->bootstrap_null();

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

	/* perform test: compute p-value and test if null-hypothesis is rejected for
	 * a test level of 0.05 */
	float64_t alpha=0.05;

	mmd->set_null_approximation_method(BOOTSTRAP);
	mmd->set_statistic_type(BIASED);
	mmd->set_bootstrap_iterations(3);
	mmd->set_num_eigenvalues_spectrum(3);
	mmd->set_num_samples_sepctrum(250);

	/* compute tpye I and II error using normal and precomputed kernel */
	index_t num_trials=3;
	SGVector<float64_t> type_I_mmds(num_trials);
	SGVector<float64_t> type_I_threshs_boot(num_trials);
	SGVector<float64_t> type_I_threshs_spectrum(num_trials);
	SGVector<float64_t> type_I_threshs_gamma(num_trials);
	SGVector<float64_t> type_II_mmds(num_trials);
	SGVector<float64_t> type_II_threshs_boot(num_trials);
	SGVector<float64_t> type_II_threshs_spectrum(num_trials);
	SGVector<float64_t> type_II_threshs_gamma(num_trials);

	SGVector<index_t> inds(2*m);
	inds.range_fill();
	CFeatures* p_and_q=mmd->get_p_and_q();

	/* use fixed seed */
	CMath::init_random(1);
	for (index_t i=0; i<num_trials; ++i)
	{
		/* this effectively means that p=q - rejecting is tpye I error */
		inds.permute();
		p_and_q->add_subset(inds);
		type_I_mmds[i]=mmd->compute_statistic();
		mmd->set_null_approximation_method(BOOTSTRAP);
		type_I_threshs_boot[i]=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_SPECTRUM);
		type_I_threshs_spectrum[i]=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_GAMMA);
		type_I_threshs_gamma[i]=mmd->compute_threshold(alpha);
		p_and_q->remove_subset();

		type_II_mmds[i]=mmd->compute_statistic();
		mmd->set_null_approximation_method(BOOTSTRAP);
		type_II_threshs_boot[i]=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_SPECTRUM);
		type_II_threshs_spectrum[i]=mmd->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_GAMMA);
		type_II_threshs_gamma[i]=mmd->compute_threshold(alpha);

	}
	SG_UNREF(p_and_q);

	//SG_SPRINT("normal kernel\n");
	//type_I_mmds.display_vector("type_I_mmds");
	//type_I_threshs_boot.display_vector("type_I_threshs_boot");
	//type_II_mmds.display_vector("type_II_mmds");
	//type_II_threshs_boot.display_vector("type_II_threshs_boot");

	/* same thing with precomputed kernel */
	SGVector<float64_t> type_I_mmds_pre(num_trials);
	SGVector<float64_t> type_I_threshs_boot_pre(num_trials);
	SGVector<float64_t> type_I_threshs_spectrum_pre(num_trials);
	SGVector<float64_t> type_I_threshs_gamma_pre(num_trials);
	SGVector<float64_t> type_II_mmds_pre(num_trials);
	SGVector<float64_t> type_II_threshs_boot_pre(num_trials);
	SGVector<float64_t> type_II_threshs_spectrum_pre(num_trials);
	SGVector<float64_t> type_II_threshs_gamma_pre(num_trials);
	kernel->init(p_and_q, p_and_q);
	CCustomKernel* precomputed=new CCustomKernel(kernel);
	CQuadraticTimeMMD* mmd2=new CQuadraticTimeMMD(precomputed, m);
	mmd2->set_null_approximation_method(BOOTSTRAP);
	mmd2->set_bootstrap_iterations(3);
	inds.range_fill();
	CMath::init_random(1);
	for (index_t i=0; i<num_trials; ++i)
	{
		/* this effectively means that p=q - rejecting is tpye I error */
		inds.permute();
		precomputed->add_row_subset(inds);
		precomputed->add_col_subset(inds);
		type_I_mmds_pre[i]=mmd2->compute_statistic();
		mmd->set_null_approximation_method(BOOTSTRAP);
		type_I_threshs_boot_pre[i]=mmd2->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_SPECTRUM);
		type_I_threshs_spectrum_pre[i]=mmd2->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_GAMMA);
		type_I_threshs_gamma_pre[i]=mmd2->compute_threshold(alpha);
		precomputed->remove_row_subset();
		precomputed->remove_col_subset();

		type_II_mmds_pre[i]=mmd2->compute_statistic();
		mmd->set_null_approximation_method(BOOTSTRAP);
		type_II_threshs_boot_pre[i]=mmd2->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_SPECTRUM);
		type_II_threshs_spectrum_pre[i]=mmd2->compute_threshold(alpha);
		mmd->set_null_approximation_method(MMD2_GAMMA);
		type_II_threshs_gamma_pre[i]=mmd2->compute_threshold(alpha);

	}

	//SG_SPRINT("precomputed kernel\n");
	//type_I_mmds_pre.display_vector("type_I_mmds");
	//type_I_threshs_boot_pre.display_vector("type_I_threshs_boot");
	//type_II_mmds_pre.display_vector("type_II_mmds");
	//type_II_threshs_boot_pre.display_vector("type_II_threshs_boot");

	for (index_t i=0; i<num_trials; ++i)
	{
		EXPECT_LE(CMath::abs(type_I_mmds_pre[i]-type_I_mmds_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_I_threshs_boot_pre[i]-type_I_threshs_boot_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_I_threshs_spectrum_pre[i]-type_I_threshs_spectrum_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_I_threshs_gamma_pre[i]-type_I_threshs_gamma_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_II_mmds_pre[i]-type_II_mmds_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_II_threshs_boot_pre[i]-type_II_threshs_boot_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_II_threshs_spectrum_pre[i]-type_II_threshs_spectrum_pre[i]), 10E-15);
		EXPECT_LE(CMath::abs(type_II_threshs_gamma_pre[i]-type_II_threshs_gamma_pre[i]), 10E-15);
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
