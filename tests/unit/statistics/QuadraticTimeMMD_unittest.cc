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
	float64_t difference=statistic-0.357650929735592;
	EXPECT_LE(CMath::abs(difference), 10E-15);

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
	CMath::init_random(1);
	SGVector<float64_t> null_samples=mmd->bootstrap_null();

	float64_t mean=CStatistics::mean(null_samples);
	float64_t var=CStatistics::variance(null_samples);

	//SG_SPRINT("mean %f\n", mean);
	//SG_SPRINT("var %f\n", var);

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
	CMath::init_random(1);
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
