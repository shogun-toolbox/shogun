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
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;


void create_mean_data(SGMatrix<float64_t> target, float64_t difference)
{
	/* create data matrix for P and Q. P is a standard normal, Q is the same but
	 * has a mean difference in one dimension */
	for (index_t i=0; i<target.num_rows; ++i)
	{
		for (index_t j=0; j<target.num_cols/2; ++j)
			target(i,j)=CMath::randn_double();

		/* add mean difference in first dimension of second half of data */
		for (index_t j=target.num_cols/2; j<target.num_cols; ++j)
				target(i,j)=CMath::randn_double() + (i==0 ? difference : 0);
	}
}

/** tests the linear mmd statistic for a single data case and ensures
 * equality with matlab implementation */
void test_linear_mmd_fixed()
{
	index_t m=2;
	index_t d=3;
	float64_t sigma=2;
	float64_t sq_sigma_twice=sigma*sigma*2;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(10, sq_sigma_twice);
	kernel->init(features, features);

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, features, m);

	/* assert matlab result */
	float64_t difference=mmd->compute_statistic()-0.034218118311602;
	ASSERT(CMath::abs(difference)<10E-16);

	SG_UNREF(mmd);
}

/** tests the linear mmd statistic for a random data case (fixed distribution
 * and ensures equality with matlab implementation */
void test_linear_mmd_random()
{
	index_t dimension=3;
	index_t m=10000;
	float64_t difference=0.5;
	float64_t sigma=2;

	index_t num_runs=100;
	SGVector<float64_t> mmds(num_runs);

	SGMatrix<float64_t> data(dimension, 2*m);
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel=new CGaussianKernel(100, sigma*sigma*2);

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, features, m);

	for (index_t i=0; i<num_runs; ++i)
	{
		create_mean_data(data, difference);
		mmds[i]=mmd->compute_statistic();
	}

	float64_t mean=CStatistics::mean(mmds);
	float64_t var=CStatistics::variance(mmds);

	/* MATLAB 100-run 3 sigma interval for mean is
	 * [ 0.006291248839741, 0.039143028479036] */
	ASSERT(mean>0.006291248839741);
	ASSERT(mean<0.039143028479036);

	/* MATLAB 100-run variance is 2.997887292969012e-05 quite stable */
	ASSERT(CMath::abs(var-2.997887292969012e-05)<10E-5);
	SG_SPRINT("%f\n", var);

	SG_UNREF(mmd);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	test_linear_mmd_fixed();
	test_linear_mmd_random();

	exit_shogun();
	return 0;
}


