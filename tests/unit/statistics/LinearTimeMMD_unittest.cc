/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 * Written (W) 2014 Soumyajit De
 */

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

/** tests the linear mmd statistic for a single data case and ensures
 * equality with python implementation. Since data from memory is used,
 * this is rather complicated, i.e. create dense features and then create
 * streaming dense features from them. Normally, just use streaming features
 * directly. */
TEST(LinearTimeMMD, statistic_single_kernel_fixed_unbiased)
{
	index_t m=2;
	index_t d=3;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* assert python result */
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.051325806508381, 1E-15);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD, statistic_single_kernel_fixed_unbiased_DEPRECATED)
{
	index_t m=2;
	index_t d=3;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);
	mmd->set_statistic_type(S_INCOMPLETE_DEPRECATED);
	mmd->set_null_var_est_method(NO_PERMUTATION_DEPRECATED);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* assert matlab result */
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.034218118311602, 1E-15);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD, statistic_single_kernel_fixed_incomplete)
{
	index_t m=2;
	index_t d=3;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);
	mmd->set_statistic_type(S_INCOMPLETE);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* assert python result */
	float64_t statistic=mmd->compute_statistic();
	EXPECT_NEAR(statistic, 0.034218118311602, 1E-15);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD, p_value_gaussian_large_num_samples)
{
	index_t m=10000;
	index_t d=2;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;

	// test power
	float64_t alpha=0.05;

	// use fixed seed for reproducibility
	CMath::init_random(1);

	// generate data from N(0,1) and N(mu,1)
	float64_t mu=0.5;
	SGMatrix<float64_t> data(d, 2*m);

	for (index_t i=0; i<d*m; ++i)
		data.matrix[i]=CMath::randn_double();

	for (index_t i=d*m; i<2*d*m; ++i)
		data.matrix[i]=CMath::randn_double()+mu;

	// create data matrix for each features (appended is not supported)
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	// create stremaing features from dense features
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	// shoguns kernel width is different
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	// create MMD instance with default blocksize 4
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);

	// use Gaussian approximation for null distribution
	mmd->set_null_approximation_method(MMD1_GAUSSIAN);

	// start streaming features parser
	streaming_p->start_parser();
	streaming_q->start_parser();

	// assert local machine computed result
	float64_t p_value=mmd->perform_test();
	EXPECT_NEAR(p_value, 0.0, 1E-15);
	EXPECT_TRUE(p_value<alpha);

	// stop streaming features parser
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD, statistic_and_Q_fixed)
{
	index_t m=8;
	index_t d=3;

	/* use fixed seed for reproducibility */
	CMath::init_random(1);

	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p_1=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q_1=new CStreamingDenseFeatures<float64_t>(
			features_q);
	CStreamingFeatures* streaming_p_2=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q_2=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* create combined kernel with values 2^5 to 2^7 */
	CCombinedKernel* kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance, using default blocksize which is 4 */
	CLinearTimeMMD* mmd_1=new CLinearTimeMMD(kernel, streaming_p_1,
			streaming_q_1, m);
	CLinearTimeMMD* mmd_2=new CLinearTimeMMD(kernel, streaming_p_2,
			streaming_q_2, m);

	/* start streaming features parser */
	streaming_p_1->start_parser();
	streaming_q_1->start_parser();
	streaming_p_2->start_parser();
	streaming_q_2->start_parser();

	/* test method */
	SGVector<float64_t> mmds_1;
	SGMatrix<float64_t> Q;
	mmd_1->compute_statistic_and_Q(mmds_1, Q);
	SGVector<float64_t> mmds_2=mmd_2->compute_statistic(true);

	/* assert that both MMD methods give the same results */
	EXPECT_EQ(mmds_1.vlen, mmds_2.vlen);
	for (index_t i=0; i<mmds_1.vlen; ++i)
		EXPECT_NEAR(mmds_1[i], mmds_2[i], 1E-15);

	/* assert actual result against fixed python code */
//	1.0e-03 *
//	   0.482892712133
//	   0.120736411855
//	   0.030184930162
	EXPECT_NEAR(mmds_1[0], 0.000482892712133, 1E-15);
	EXPECT_NEAR(mmds_1[1], 0.000120736411855, 1E-15);
	EXPECT_NEAR(mmds_1[2], 0.000030184930162, 1E-15);

	/* assert correctness of Q matrix */
	EXPECT_NEAR(Q(0,0), 1.7396757355940154e-08, 1E-15);
	EXPECT_NEAR(Q(1,0), 1.9697427274323797e-08, 1E-15);
	EXPECT_NEAR(Q(2,0), 5.2746537284212936e-09, 1E-15);
	EXPECT_NEAR(Q(0,1), 1.9697427274323797e-08, 1E-15);
	EXPECT_NEAR(Q(1,1), 4.9251782982851156e-09, 1E-15);
	EXPECT_NEAR(Q(2,1), 1.3188798945699736e-09, 1E-15);
	EXPECT_NEAR(Q(0,2), 5.2746537284212936e-09, 1E-15);
	EXPECT_NEAR(Q(1,2), 1.3188798945699736e-09, 1E-15);
	EXPECT_NEAR(Q(2,2), 3.2973350458817279e-10, 1E-15);

	/* start streaming features parser */
	streaming_p_1->end_parser();
	streaming_q_1->end_parser();
	streaming_p_2->end_parser();
	streaming_q_2->end_parser();

	SG_UNREF(mmd_1);
	SG_UNREF(mmd_2);
}

TEST(LinearTimeMMD, statistic_and_Q_fixed_DEPRECATED)
{
	index_t m=8;
	index_t d=3;

	/* use fixed seed for reproducibility */
	CMath::init_random(1);

	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p_1=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q_1=new CStreamingDenseFeatures<float64_t>(
			features_q);
	CStreamingFeatures* streaming_p_2=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q_2=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* create combined kernel with values 2^5 to 2^7 */
	CCombinedKernel* kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance, using default blocksize which is 4 */
	CLinearTimeMMD* mmd_1=new CLinearTimeMMD(kernel, streaming_p_1,
			streaming_q_1, m);
	CLinearTimeMMD* mmd_2=new CLinearTimeMMD(kernel, streaming_p_2,
			streaming_q_2, m);

	mmd_1->set_null_var_est_method(NO_PERMUTATION_DEPRECATED);
	mmd_2->set_null_var_est_method(NO_PERMUTATION_DEPRECATED);

	/* start streaming features parser */
	streaming_p_1->start_parser();
	streaming_q_1->start_parser();
	streaming_p_2->start_parser();
	streaming_q_2->start_parser();

	/* test method */
	SGVector<float64_t> mmds_1;
	SGMatrix<float64_t> Q;
	mmd_1->compute_statistic_and_Q(mmds_1, Q);
	SGVector<float64_t> mmds_2=mmd_2->compute_statistic(true);

	/* assert that both MMD methods give the same results */
	EXPECT_EQ(mmds_1.vlen, mmds_2.vlen);
	for (index_t i=0; i<mmds_1.vlen; ++i)
		EXPECT_NEAR(mmds_1[i], mmds_2[i], 1E-15);

	/* assert actual result against fixed python code */
//	1.0e-03 *
//	   0.482892712133
//	   0.120736411855
//	   0.030184930162
	EXPECT_NEAR(mmds_1[0], 0.000482892712133, 1E-15);
	EXPECT_NEAR(mmds_1[1], 0.000120736411855, 1E-15);
	EXPECT_NEAR(mmds_1[2], 0.000030184930162, 1E-15);

	/* assert correctness of Q matrix with local machine computed values */
	EXPECT_NEAR(Q(0,0), 1.6960665492375955e-07, 1E-15);
	EXPECT_NEAR(Q(1,0), 4.2407259823893387e-08, 1E-15);
	EXPECT_NEAR(Q(2,0), 1.0602164750786324e-08, 1E-15);
	EXPECT_NEAR(Q(0,1), 4.2407259823893387e-08, 1E-15);
	EXPECT_NEAR(Q(1,1), 1.0603214175123433e-08, 1E-15);
	EXPECT_NEAR(Q(2,1), 2.6508910047299925e-09, 1E-15);
	EXPECT_NEAR(Q(0,2), 1.0602164750786324e-08, 1E-15);
	EXPECT_NEAR(Q(1,2), 2.6508910047299925e-09, 1E-15);
	EXPECT_NEAR(Q(2,2), 6.627446171852379e-10, 1E-15);

	/* start streaming features parser */
	streaming_p_1->end_parser();
	streaming_q_1->end_parser();
	streaming_p_2->end_parser();
	streaming_q_2->end_parser();

	SG_UNREF(mmd_1);
	SG_UNREF(mmd_2);
}

TEST(LinearTimeMMD, statistic_and_variance_multiple_kernels_fixed_same_num_samples)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* create combined kernel with values 2^5 to 2^7 */
	CCombinedKernel* kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);

	/* using within-block direct estimation for asserting results */
	mmd->set_null_var_est_method(WITHIN_BLOCK_DIRECT);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* test method */
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	mmd->compute_statistic_and_variance(mmds, vars, true);

	/* assert actual result against fixed python code */
//	1.0e-03 *
//	   0.482892712133
//	   0.120736411855
//	   0.030184930162
	EXPECT_NEAR(mmds[0], 0.000482892712133, 1E-15);
	EXPECT_NEAR(mmds[1], 0.000120736411855, 1E-15);
	EXPECT_NEAR(mmds[2], 0.000030184930162, 1E-15);

	/* assert correctness of variance estimates */
//	vars =
//	   1.0e-08 *
//	   3.7022768
//	   0.2314493
//	   0.0144666
	EXPECT_NEAR(vars[0], 0.000000037022768, 1E-14);
	EXPECT_NEAR(vars[1], 0.000000002314493, 1E-14);
	EXPECT_NEAR(vars[2], 0.000000000144666, 1E-14);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD, statistic_and_variance_multiple_kernels_fixed_DEPRECATED)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d, 2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* create combined kernel with values 2^5 to 2^7 */
	CCombinedKernel* kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);
	mmd->set_statistic_type(S_INCOMPLETE_DEPRECATED);
	mmd->set_null_var_est_method(NO_PERMUTATION_DEPRECATED);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* test method */
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	mmd->compute_statistic_and_variance(mmds, vars, true);

	EXPECT_NEAR(mmds[0], 0.00025363731337291195, 1E-15);
	EXPECT_NEAR(mmds[1], 0.00006341684175759088, 1E-15);
	EXPECT_NEAR(mmds[2], 0.00001585468008052926, 1E-15);

	EXPECT_NEAR(vars[0], 1.7441397282197432e-08, 1E-14);
	EXPECT_NEAR(vars[1], 1.090413580583456e-09, 1E-14);
	EXPECT_NEAR(vars[2], 6.8155947599845925e-11, 1E-14);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD, statistic_and_variance_fixed_multiple_kernels_different_num_samples)
{
	index_t m=8;
	index_t n=12;
	index_t d=3;
	SGMatrix<float64_t> data(d, m+n);
	for (index_t i=0; i<d*(m+n); ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, n);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*n);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	for (index_t i=0; i<d*m; ++i)
		data_p.matrix[i]/=max_p;

	for (index_t i=0; i<d*n; ++i)
		data_q.matrix[i]/=max_q;

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=new CStreamingDenseFeatures<float64_t>(
			features_p);
	CStreamingFeatures* streaming_q=new CStreamingDenseFeatures<float64_t>(
			features_q);

	/* create combined kernel with values 2^5 to 2^7 */
	CCombinedKernel* kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance using blocksize 5 (2 from p and 3 from q per block) */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m, n, 5);

	/* using within-block direct estimation for asserting results */
	mmd->set_null_var_est_method(WITHIN_BLOCK_DIRECT);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* test method */
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	mmd->compute_statistic_and_variance(mmds, vars, true);

	/* assert actual result against fixed python code */
	EXPECT_NEAR(mmds[0], 0.000329816772926, 1E-15);
	EXPECT_NEAR(mmds[1], 0.000082460957462, 1E-15);
	EXPECT_NEAR(mmds[2], 0.000020615662162, 1E-15);

	/* assert correctness of variance estimates */
	EXPECT_NEAR(vars[0], 0.000000009758379, 1E-14);
	EXPECT_NEAR(vars[1], 0.000000000610013, 1E-14);
	EXPECT_NEAR(vars[2], 0.000000000038129, 1E-14);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}
