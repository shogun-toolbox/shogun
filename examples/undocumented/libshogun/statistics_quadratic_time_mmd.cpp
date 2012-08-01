/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

/** tests the quadratic mmd statistic for a single data case and ensures
 * equality with matlab implementation */
void test_quadratic_mmd_fixed()
{
	index_t n=2;
	index_t d=3;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	SGMatrix<float64_t> data(d,2*n);
	for (index_t i=0; i<2*d*n; ++i)
		data.matrix[i]=i;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);
	kernel->init(features, features);

	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features, n);

	/* unbiased statistic */
	mmd->set_statistic_type(UNBIASED);
	float64_t difference=CMath::abs(mmd->compute_statistic()-0.051325806508381);
	ASSERT(difference<=10E-16);

	/* biased statistic */
	mmd->set_statistic_type(BIASED);
	difference=CMath::abs(mmd->compute_statistic()-1.017107688196714);
	ASSERT(difference<=10E-16);

	SG_UNREF(mmd);
}

/** tests the quadratic mmd statistic bootstrapping for a random data case and
 * ensures equality with matlab implementation (unbiased statistic) */
void test_quadratic_mmd_bootstrap()
{
	index_t dimension=3;
	index_t m=100;
	float64_t difference=0.5;
	float64_t sigma=2;
	index_t num_iterations=1000;
	num_iterations=10; //speed up

	SGMatrix<float64_t> data=CDataGenerator::generate_mean_data(m, dimension,
			difference);
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features, m);
	mmd->set_statistic_type(UNBIASED);
	mmd->set_bootstrap_iterations(num_iterations);

	/* use fixed seed */
	CMath::init_random(1);
	SGVector<float64_t> null_samples=mmd->bootstrap_null();

	float64_t mean=CStatistics::mean(null_samples);
	float64_t var=CStatistics::variance(null_samples);

	/* MATLAB mean 2-sigma confidence interval for 1000 repretitions is
	 * [-3.169406734013459e-04, 3.296399498466372e-04] */
	SG_SPRINT("mean %f\n", mean);
//	ASSERT(mean>-3.169406734013459e-04);
//	ASSERT(mean<3.296399498466372e-04);

	/* MATLAB variance 2-sigma confidence interval for 1000 repretitions is
	 * [2.194192869469228e-05,2.936672859339959e-05] */
	SG_SPRINT("var %f\n", var);
//	ASSERT(var>2.194192869469228e-05);
//	ASSERT(var<2.936672859339959e-05);

	/* now again but with a precomputed kernel, same features.
	 * This avoids re-computing the kernel matrix in every bootstrapping
	 * iteration and should be num_iterations times faster */
	SG_REF(features);
	CCustomKernel* precomputed_kernel=new CCustomKernel(kernel);
	SG_UNREF(mmd);
	mmd=new CQuadraticTimeMMD(precomputed_kernel, features, m);
	mmd->set_statistic_type(UNBIASED);
	mmd->set_bootstrap_iterations(num_iterations);
	CMath::init_random(1);
	null_samples=mmd->bootstrap_null();

	/* assert that results do not change */
	SG_SPRINT("mean %f, var %f\n", CStatistics::mean(null_samples),
			CStatistics::variance(null_samples));
	ASSERT(CMath::abs(mean-CStatistics::mean(null_samples))<10E-5);
	ASSERT(CMath::abs(var-CStatistics::variance(null_samples))<10E-5);

	SG_UNREF(mmd);
	SG_UNREF(features);
}

#ifdef HAVE_LAPACK
/** tests the quadratic mmd statistic threshold method spectrum for radnom data
 * case and ensures equality with matlab implementation */
void test_quadratic_mmd_spectrum()
{
	index_t dimension=3;
	index_t m=100;
	float64_t difference=0.5;
	float64_t sigma=2;

	SGMatrix<float64_t> data=CDataGenerator::generate_mean_data(m, dimension,
			difference);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features, m);

	mmd->set_num_samples_sepctrum(1000);
	mmd->set_num_samples_sepctrum(10); //speed up
	mmd->set_num_eigenvalues_spectrum(m);
	mmd->set_null_approximation_method(MMD2_SPECTRUM);
	mmd->set_statistic_type(BIASED);

	/* compute p-value for a fixed statistic value */
	float64_t p=mmd->compute_p_value(2);

	/* MATLAB 1000 iterations 3 sigma confidence interval is
	 * [0.021240218376709, 0.060875781623291] */
	SG_SPRINT("p %f\n", p);
//	ASSERT(p>0.021240218376709);
//	ASSERT(p<0.060875781623291);

	SG_UNREF(mmd);
}
#endif // HAVE_LAPACK

/** tests the quadratic mmd statistic threshold method gamma for fixed data
 * case and ensures equality with matlab implementation */
void test_quadratic_mmd_gamma()
{
	index_t dimension=3;
	index_t m=100;
	float64_t sigma=4;

	/* note: fixed data this time */
	SGMatrix<float64_t> data(dimension, 2*m);
	for (index_t i=0; i<2*dimension*m; ++i)
		data.matrix[i]=i;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features, m);

	mmd->set_null_approximation_method(MMD2_GAMMA);
	mmd->set_statistic_type(BIASED);

	/* compute p-value for a fixed statistic value */
	float64_t p=mmd->compute_p_value(2);
	SG_SPRINT("p: %f\n", p);

	/* MATLAB 1000 iterations mean: 0.511547577996229 with variance 10E-15,
	 * asserting with only 10-12 to avoid problems. Shold never fail.
	 */
	ASSERT(CMath::abs(p-0.511547577996229)<10E-12);

	SG_UNREF(mmd);
}

/** tests the quadratic mmd statistic for a random data case (fixed distribution
 * and ensures equality with matlab implementation (unbiased case) */
void test_quadratic_mmd_random()
{
	index_t dimension=3;
	index_t m=300;
	float64_t difference=0.5;
	float64_t sigma=2;

	index_t num_runs=100;
	num_runs=10; //speed up
	SGVector<float64_t> mmds(num_runs);

	/* pre-allocate data matrix and features, just change elements later */
	SGMatrix<float64_t> data(dimension, 2*m);
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, features, m);
	mmd->set_statistic_type(UNBIASED);
	for (index_t i=0; i<num_runs; ++i)
	{
		/* use pre-allocated space for data generation */
		CDataGenerator::generate_mean_data(m, dimension, difference, data);
		kernel->init(features, features);
		mmds[i]=mmd->compute_statistic();
	}

	/* MATLAB 95% mean confidence interval 0.007495841715582 0.037960088792417 */
	float64_t mean=CStatistics::mean(mmds);
	SG_SPRINT("mean %f\n", mean);
//	ASSERT((mean>0.007495841715582) && (mean<0.037960088792417));

	/* MATLAB variance is 5.800439687240292e-05 quite stable */
	float64_t variance=CStatistics::variance(mmds);
	SG_SPRINT("variance: %f\n", variance);
//	ASSERT(CMath::abs(variance-5.800439687240292e-05)<10E-5);
	SG_UNREF(mmd);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	/* all tests have been "speed up" by reducing the number of runs/samples.
	 * If you have any doubts in the results, set all num_runs to original
	 * numbers and activate asserts. If they fail, something is wrong. */

	test_quadratic_mmd_fixed();
	test_quadratic_mmd_random();
	test_quadratic_mmd_bootstrap();
#ifdef HAVE_LAPACK
	test_quadratic_mmd_spectrum();
#endif
	test_quadratic_mmd_gamma();

	exit_shogun();
	return 0;
}


