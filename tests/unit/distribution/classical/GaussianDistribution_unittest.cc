/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifdef HAVE_EIGEN3

#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(GaussianDistribution,log_pdf_single_1d)
{
	SGVector<float64_t> mean(1);
	SGMatrix<float64_t> cov(1,1);
	mean[0]=1;
	cov(0,0)=2;
	CGaussianDistribution* gauss=new CGaussianDistribution(mean,cov);

	SGVector<float64_t> x(1);
	x[0]=0;
	float64_t result=((CProbabilityDistribution*)gauss)->log_pdf(x);

	EXPECT_NEAR(result, -1.5155121234846454, 1e-15);

	SG_UNREF(gauss);
}

TEST(GaussianDistribution,log_pdf_multiple_1d)
{
	SGVector<float64_t> mean(1);
	SGMatrix<float64_t> cov(1,1);
	mean[0]=1;
	cov(0,0)=2;
	CGaussianDistribution* gauss=new CGaussianDistribution(mean,cov);

	SGMatrix<float64_t> x(1,2);
	x(0,0)=0;
	x(0,1)=1;
	SGVector<float64_t> result=(gauss)->log_pdf_multiple(x);

	EXPECT_NEAR(result[0], -1.5155121234846454, 1e-15);
	EXPECT_NEAR(result[1], -1.2655121234846454, 1e-15);

	SG_UNREF(gauss);
}

TEST(GaussianDistribution,log_pdf_single_2d)
{
	SGVector<float64_t> mean(2);
	SGMatrix<float64_t> cov(2,2);
	mean[0]=1;
	mean[1]=2;
	cov(0,0)=2.4;
	cov(0,1)=1.3;
	cov(1,0)=1.3;
	cov(1,1)=2.4;
	CGaussianDistribution* gauss=new CGaussianDistribution(mean,cov);

	SGVector<float64_t> x(2);
	x[0]=0;
	x[1]=0;
	float64_t result=((CProbabilityDistribution*)gauss)->log_pdf(x);

	EXPECT_NEAR(result, -3.375079401517433, 1e-15);

	SG_UNREF(gauss);
}

TEST(GaussianDistribution,log_pdf_multiple_2d)
{
	SGVector<float64_t> mean(2);
	SGMatrix<float64_t> cov(2,2);
	mean[0]=1;
	mean[1]=2;
	cov(0,0)=2.4;
	cov(0,1)=1.3;
	cov(1,0)=1.3;
	cov(1,1)=2.4;
	CGaussianDistribution* gauss=new CGaussianDistribution(mean,cov);

	SGMatrix<float64_t> x(2,2);
	x(0,0)=1;
	x(1,0)=2;
	x(0,1)=3;
	x(1,1)=4;
	SGVector<float64_t> result=(gauss)->log_pdf_multiple(x);

	EXPECT_NEAR(result[0], -2.539698566136597, 1e-15);
	EXPECT_NEAR(result[1], -3.620779647217678, 1e-15);

	SG_UNREF(gauss);
}

TEST(GaussianDistribution,sample_2d_fixed)
{
	SGVector<float64_t> mean(2);
	SGMatrix<float64_t> cov(2,2);
	mean[0]=1;
	mean[1]=2;
	cov(0,0)=2.4;
	cov(0,1)=1.3;
	cov(1,0)=1.3;
	cov(1,1)=2.4;
	CGaussianDistribution* gauss=new CGaussianDistribution(mean,cov);

	/* fake std normal samples */
	SGMatrix<float64_t> pre_samples(2,2);
	pre_samples(0,0)=-1.93251186;
	pre_samples(1,0)=1.64881715;
	pre_samples(0,1)=0.44701692;
	pre_samples(1,1)=-1.17987856;
	SGMatrix<float64_t> result=gauss->sample(pre_samples.num_cols, pre_samples);

	EXPECT_NEAR(result(0,0), -1.9938345, 1e-7);
	EXPECT_NEAR(result(0,1), 1.69251563, 1e-7);
	EXPECT_NEAR(result(1,0), 2.52549802, 1e-7);
	EXPECT_NEAR(result(1,1), 0.83862562, 1e-7);


	SG_UNREF(gauss);
}

TEST(GaussianDistribution,sample_2d)
{
	SGVector<float64_t> mean(2);
	SGMatrix<float64_t> cov(2,2);
	mean[0]=1;
	mean[1]=2;
	cov(0,0)=2.4;
	cov(0,1)=1.3;
	cov(1,0)=1.3;
	cov(1,1)=2.4;
	CGaussianDistribution* gauss=new CGaussianDistribution(mean,cov);

	index_t num_samples=100000;
	SGMatrix<float64_t> samples=gauss->sample(num_samples);
	SGMatrix<float64_t> emp_cov(2,2);
	Map<MatrixXd> eigen_samples(samples.matrix, samples.num_rows, samples.num_cols);
	Map<MatrixXd> eigen_emp_cov(emp_cov.matrix, emp_cov.num_rows, emp_cov.num_cols);

	/* center and compute empirical covariance */
	MatrixXd centered = eigen_samples.colwise() - eigen_samples.rowwise().mean();
	eigen_emp_cov = 1.0/(num_samples-1)*(centered * centered.transpose());

	for (index_t i=0; i<cov.num_rows*cov.num_cols; ++i)
		EXPECT_NEAR(cov.matrix[i], emp_cov.matrix[i], 1e-1);

	SG_UNREF(gauss);
}

#endif // HAVE_EIGEN3
