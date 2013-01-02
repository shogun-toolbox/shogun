/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

/** tests the linear mmd statistic for a single data case and ensures
 * equality with matlab implementation. Since data from memory is used,
 * this is rather complicated, i.e. create dense features and then create
 * streaming dense features from them. Normally, just use streaming features
 * directly. */
void test_linear_mmd_fixed()
{
	index_t m=2;
	index_t d=3;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	SGMatrix<float64_t> data(d,2*m);
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
	CStreamingFeatures* streaming_p=
			new CStreamingDenseFeatures<float64_t>(features_p);
	CStreamingFeatures* streaming_q=
			new CStreamingDenseFeatures<float64_t>(features_q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p, streaming_q, m);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* assert matlab result */
	float64_t statistic=mmd->compute_statistic();
	SG_SPRINT("statistic=%f\n", statistic);
	float64_t difference=statistic-0.034218118311602;
	ASSERT(CMath::abs(difference)<10E-16);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

/** tests the linear mmd statistic for a random data case (fixed distribution
 * and ensures equality with matlab implementation */
void test_linear_mmd_random()
{
	index_t d=3;
	index_t m=10000;
	float64_t difference=0.5;
	float64_t sigma=2;

	index_t num_runs=100;
	num_runs=3; //speed up
	SGVector<float64_t> mmds(num_runs);

	/* create data generator classes that implement a meanshift in q */
	CMeanShiftDataGenerator<float64_t>* gen_p=
			new CMeanShiftDataGenerator<float64_t>(0, d);
	CMeanShiftDataGenerator<float64_t>* gen_q=
			new CMeanShiftDataGenerator<float64_t>(difference, d);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, gen_p, gen_q, m);

	/* start parser of streaming features */
	gen_p->start_parser();
	gen_q->start_parser();

	/* compute statistic streams new data all the time */
	for (index_t i=0; i<num_runs; ++i)
		mmds[i]=mmd->compute_statistic();

	/* stop parser of streaming features */
	gen_p->end_parser();
	gen_q->end_parser();

	float64_t mean=CStatistics::mean(mmds);
	float64_t var=CStatistics::variance(mmds);

	SG_SPRINT("mean %f\n", mean);
	SG_SPRINT("var %f\n", var);

	SG_UNREF(mmd);
}

void test_linear_mmd_variance_estimate()
{
	index_t d=3;
	index_t m=10000;
	float64_t difference=0.5;
	float64_t sigma=2;

	index_t num_runs=100;
	num_runs=10; //speed up
	SGVector<float64_t> vars(num_runs);

	/* create data generator classes that implement a meanshift in q */
	CMeanShiftDataGenerator<float64_t>* gen_p=
			new CMeanShiftDataGenerator<float64_t>(0, d);
	CMeanShiftDataGenerator<float64_t>* gen_q=
			new CMeanShiftDataGenerator<float64_t>(difference, d);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, gen_p, gen_q, m);

	/* start parser of streaming features */
	gen_p->start_parser();
	gen_q->start_parser();

	for (index_t i=0; i<num_runs; ++i)
		vars[i]=mmd->compute_variance_estimate();

	/* stop parser of streaming features */
	gen_p->end_parser();
	gen_q->end_parser();

	float64_t mean=CStatistics::mean(vars);
	float64_t var=CStatistics::variance(vars);

	/* MATLAB 100-run 3 sigma interval for mean is
	 * [2.487949168976897e-05, 2.816652377191562e-05] */
	SG_SPRINT("mean variance %f\n", mean);
//	ASSERT(mean>2.487949168976897e-05);
//	ASSERT(mean<2.816652377191562e-05);

	/* MATLAB 100-run variance is  8.321246145460274e-06 quite stable */
	SG_SPRINT("var of variance %f\n", var);
	ASSERT(CMath::abs(var-8.321246145460274e-06)<10E-6);

	SG_UNREF(mmd);
}

void test_linear_mmd_variance_estimate_vs_bootstrap()
{
	index_t d=3;
	index_t m=50000;
	m=1000; //speed up
	float64_t difference=0.5;
	float64_t sigma=2;

	/* create data generator classes that implement a meanshift in q */
	CMeanShiftDataGenerator<float64_t>* gen_p=
			new CMeanShiftDataGenerator<float64_t>(0, d);
	CMeanShiftDataGenerator<float64_t>* gen_q=
			new CMeanShiftDataGenerator<float64_t>(difference, d);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, gen_p, gen_q, m);

	/* start parser of streaming features */
	gen_p->start_parser();
	gen_q->start_parser();

	/* for checking results, set to 100 */
	mmd->set_bootstrap_iterations(100);
	mmd->set_bootstrap_iterations(100); // speed up
	SGVector<float64_t> null_samples=mmd->bootstrap_null();
	float64_t bootstrap_variance=CStatistics::variance(null_samples);
	SGVector<float64_t> statistic;
	SGVector<float64_t> estimated_variance;

	/* it is also possible to compute these separately, but this only requires
	 * one loop and values are connected */
	mmd->compute_statistic_and_variance(statistic, estimated_variance);
	float64_t variance_error=CMath::abs(bootstrap_variance-estimated_variance[0]);

	/* start parser of streaming features */
	gen_p->end_parser();
	gen_q->end_parser();

	/* assert that variances error is less than 10E-5 of statistic */
	SG_SPRINT("null distribution variance: %f\n", bootstrap_variance);
	SG_SPRINT("estimated variance: %f\n", estimated_variance[0]);
	SG_SPRINT("linear mmd itself: %f\n", statistic[0]);
	SG_SPRINT("variance error: %f\n", variance_error);
	SG_SPRINT("error/statistic: %f\n", variance_error/statistic[0]);
//	ASSERT(variance_error/statistic<10E-5);

	SG_UNREF(mmd);
}

void test_linear_mmd_type2_error()
{
	index_t d=3;
	index_t m=10000;
	float64_t difference=0.4;
	float64_t sigma=2;

	index_t num_runs=500;
	num_runs=50; // speed up
	index_t num_errors=0;

	/* create data generator classes that implement a meanshift in q */
	CMeanShiftDataGenerator<float64_t>* gen_p=
			new CMeanShiftDataGenerator<float64_t>(0, d);
	CMeanShiftDataGenerator<float64_t>* gen_q=
			new CMeanShiftDataGenerator<float64_t>(difference, d);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, gen_p, gen_q, m);
	mmd->set_null_approximation_method(MMD1_GAUSSIAN);

	for (index_t i=0; i<num_runs; ++i)
	{
		float64_t statistic=mmd->compute_statistic();
		float64_t p_value_est=mmd->compute_p_value(statistic);

		/* lets allow a 5% type 1 error */
		num_errors+=p_value_est<0.05 ? 0 : 1;
	}

	float64_t type_2_error=(float64_t)num_errors/(float64_t)num_runs;
	SG_SPRINT("type2 error est: %f\n", type_2_error);

	/* for 100 MATLAB runs, 3*sigma error range lies in
	 * [0.024568646859226, 0.222231353140774] */
//	ASSERT(type_2_error>0.024568646859226);
//	ASSERT(type_2_error<0.222231353140774);

	SG_UNREF(mmd);
}

void test_linear_mmd_statistic_and_Q_fixed()
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

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	data_p.display_matrix("data_p");
	data_q.display_matrix("data_q");

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p_1=
			new CStreamingDenseFeatures<float64_t>(features_p);
	CStreamingFeatures* streaming_q_1=
			new CStreamingDenseFeatures<float64_t>(features_q);
	CStreamingFeatures* streaming_p_2=
			new CStreamingDenseFeatures<float64_t>(features_p);
	CStreamingFeatures* streaming_q_2=
			new CStreamingDenseFeatures<float64_t>(features_q);

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
	Q.display_matrix("Q");
	mmds_1.display_vector("mmds_1");
	mmds_2.display_vector("mmds_2");

	/* assert that both MMD methods give the same results */
	ASSERT(mmds_1.vlen==mmds_2.vlen);
	for (index_t i=0; i<mmds_1.vlen; ++i)
		ASSERT(mmds_1[i]==mmds_2[i]);

	/* assert actual result against fixed MATLAB code */
//	1.0e-03 *
//	   0.156085264965383
//	   0.039043151854851
//	   0.009762153067083
	ASSERT(CMath::abs(mmds_1[0]-0.000156085264965383)<10E-18);
	ASSERT(CMath::abs(mmds_1[1]-0.000039043151854851)<10E-18);
	ASSERT(CMath::abs(mmds_1[2]-0.000009762153067083)<10E-18);

	/* assert correctness of Q matrix */
//	   1.0e-07 *
//	   0.403271337407935   0.100876370041104   0.025222752103390
//	   0.100876370041104   0.025233734937354   0.006309349164329
//	   0.025222752103390   0.006309349164329   0.001577566181822
	ASSERT(CMath::abs(Q(0,0)-0.403271337407935E-7)<10E-22);
	ASSERT(CMath::abs(Q(1,0)-0.100876370041104E-7)<10E-22);
	ASSERT(CMath::abs(Q(2,0)-0.025222752103390E-7)<10E-22);
	ASSERT(CMath::abs(Q(0,1)-0.100876370041104E-7)<10E-22);
	ASSERT(CMath::abs(Q(1,1)-0.025233734937354E-7)<10E-22);
	ASSERT(CMath::abs(Q(2,1)-0.006309349164329E-7)<10E-22);
	ASSERT(CMath::abs(Q(0,2)-0.025222752103390E-7)<10E-22);
	ASSERT(CMath::abs(Q(1,2)-0.006309349164329E-7)<10E-22);
	ASSERT(CMath::abs(Q(2,2)-0.001577566181822E-7)<10E-22);

	/* start streaming features parser */
	streaming_p_1->end_parser();
	streaming_q_1->end_parser();
	streaming_p_2->end_parser();
	streaming_q_2->end_parser();

	SG_UNREF(mmd_1);
	SG_UNREF(mmd_2);
}

void test_linear_mmd_statistic_and_variance_fixed()
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

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	data_p.display_matrix("data_p");
	data_q.display_matrix("data_q");

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=
			new CStreamingDenseFeatures<float64_t>(features_p);
	CStreamingFeatures* streaming_q=
			new CStreamingDenseFeatures<float64_t>(features_q);

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
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, streaming_p,
			streaming_q, m);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* test method */
	SGVector<float64_t> mmds;
	SGVector<float64_t> vars;
	mmd->compute_statistic_and_variance(mmds, vars, true);

	/* display results */
	vars.display_vector("vars");
	mmds.display_vector("mmds");

	/* assert actual result against fixed MATLAB code */
//	mmds=
//	1.0e-03 *
//	   0.156085264965383
//	   0.039043151854851
//	   0.009762153067083
	ASSERT(CMath::abs(mmds[0]-0.000156085264965383)<10E-18);
	ASSERT(CMath::abs(mmds[1]-0.000039043151854851)<10E-18);
	ASSERT(CMath::abs(mmds[2]-0.000009762153067083)<10E-18);

	/* assert correctness of variance estimates */
//	vars =
//	   1.0e-08 *
//	   0.418667765635434
//	   0.026197180636036
//	   0.001637799815771
	ASSERT(CMath::abs(vars[0]-0.418667765635434E-8)<10E-23);
	ASSERT(CMath::abs(vars[1]-0.026197180636036E-8)<10E-23);
	ASSERT(CMath::abs(vars[2]-0.001637799815771E-8)<10E-23);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(mmd);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
//	sg_io->set_loglevel(MSG_DEBUG);

	/* all tests have been "speed up" by reducing the number of runs/samples.
	 * If you have any doubts in the results, set all num_runs to original
	 * numbers and activate asserts. If they fail, something is likely wrong.
	 */
	test_linear_mmd_fixed();
	test_linear_mmd_random();
	test_linear_mmd_variance_estimate();
	test_linear_mmd_variance_estimate_vs_bootstrap();
	test_linear_mmd_type2_error();
	test_linear_mmd_statistic_and_Q_fixed();
	test_linear_mmd_statistic_and_variance_fixed();

	exit_shogun();
	return 0;
}

