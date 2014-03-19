/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

/** tests the linear mmd statistic for a single data case and ensures
 * equality with matlab implementation. Since data from memory is used,
 * this is rather complicated, i.e. create dense features and then create
 * streaming dense features from them. Normally, just use streaming features
 * directly. */
TEST(LinearTimeMMD,test_linear_mmd_fixed)
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

	/* assert matlab result */
	float64_t statistic=mmd->compute_statistic();
	//SG_SPRINT("statistic=%f\n", statistic);
	float64_t difference=statistic-0.034218118311602;
	EXPECT_LE(CMath::abs(difference), 10E-16);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

TEST(LinearTimeMMD,test_linear_mmd_statistic_and_Q_fixed)
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

	//SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	//data_p.display_matrix("data_p");
	//data_q.display_matrix("data_q");

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

	/* create MMD instance */
	CLinearTimeMMD* mmd_1=new CLinearTimeMMD(kernel, streaming_p_1,
			streaming_q_1, m);
	CLinearTimeMMD* mmd_2=new CLinearTimeMMD(kernel, streaming_p_2,
			streaming_q_2, m);

	/* results only equal if blocksize is larger than number of samples (other-
	 * wise, samples are processed in a different combination). In practice,
	 * just use some large value */
	mmd_1->set_blocksize(m);
	mmd_2->set_blocksize(m);

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

	/* display results */
	//Q.display_matrix("Q");
	//mmds_1.display_vector("mmds_1");
	//mmds_2.display_vector("mmds_2");

	/* assert that both MMD methods give the same results */
	EXPECT_EQ(mmds_1.vlen, mmds_2.vlen);
	for (index_t i=0; i<mmds_1.vlen; ++i)
		EXPECT_EQ(mmds_1[i], mmds_2[i]);

	/* assert actual result against fixed MATLAB code */
//	1.0e-03 *
//	   0.156085264965383
//	   0.039043151854851
//	   0.009762153067083
	EXPECT_LE(CMath::abs(mmds_1[0]-0.000156085264965383), 10E-18);
	EXPECT_LE(CMath::abs(mmds_1[1]-0.000039043151854851), 10E-18);
	EXPECT_LE(CMath::abs(mmds_1[2]-0.000009762153067083), 10E-18);

	/* assert correctness of Q matrix */
//	   1.0e-07 *
//	   0.403271337407935   0.100876370041104   0.025222752103390
//	   0.100876370041104   0.025233734937354   0.006309349164329
//	   0.025222752103390   0.006309349164329   0.001577566181822
	EXPECT_LE(CMath::abs(Q(0,0)-0.403271337407935E-7), 10E-22);
	EXPECT_LE(CMath::abs(Q(1,0)-0.100876370041104E-7), 10E-22);
	EXPECT_LE(CMath::abs(Q(2,0)-0.025222752103390E-7), 10E-22);
	EXPECT_LE(CMath::abs(Q(0,1)-0.100876370041104E-7), 10E-22);
	EXPECT_LE(CMath::abs(Q(1,1)-0.025233734937354E-7) ,10E-22);
	EXPECT_LE(CMath::abs(Q(2,1)-0.006309349164329E-7) ,10E-22);
	EXPECT_LE(CMath::abs(Q(0,2)-0.025222752103390E-7) ,10E-22);
	EXPECT_LE(CMath::abs(Q(1,2)-0.006309349164329E-7) ,10E-22);
	EXPECT_LE(CMath::abs(Q(2,2)-0.001577566181822E-7) ,10E-22);

	/* start streaming features parser */
	streaming_p_1->end_parser();
	streaming_q_1->end_parser();
	streaming_p_2->end_parser();
	streaming_q_2->end_parser();

	SG_UNREF(mmd_1);
	SG_UNREF(mmd_2);
}

TEST(LinearTimeMMD,test_linear_mmd_statistic_and_variance_fixed)
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

	//SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	//data_p.display_matrix("data_p");
	//data_q.display_matrix("data_q");

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

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* test method */
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	mmd->compute_statistic_and_variance(mmds, vars, true);

	/* display results */
	//vars.display_vector("vars");
	//mmds.display_vector("mmds");

	/* assert actual result against fixed MATLAB code */
//	mmds=
//	1.0e-03 *
//	   0.156085264965383
//	   0.039043151854851
//	   0.009762153067083
	EXPECT_LE(CMath::abs(mmds[0]-0.000156085264965383), 10E-18);
	EXPECT_LE(CMath::abs(mmds[1]-0.000039043151854851), 10E-18);
	EXPECT_LE(CMath::abs(mmds[2]-0.000009762153067083), 10E-18);

	/* assert correctness of variance estimates */
//	vars =
//	   1.0e-08 *
//	   0.418667765635434
//	   0.026197180636036
//	   0.001637799815771
	EXPECT_LE(CMath::abs(vars[0]-0.418667765635434E-8), 10E-23);
	EXPECT_LE(CMath::abs(vars[1]-0.026197180636036E-8), 10E-23);
	EXPECT_LE(CMath::abs(vars[2]-0.001637799815771E-8), 10E-23);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}
