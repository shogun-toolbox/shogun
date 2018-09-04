/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Heiko Strathmann
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
 *
 */
#include <gtest/gtest.h>

#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>


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

TEST(GaussianDistribution,univariate_log_pdf)
{
	float64_t mu, sigma2, sample;

	mu = 1.0;
	sigma2 = 1.0;
	sample = 1.0;
	EXPECT_NEAR(
	    CGaussianDistribution::univariate_log_pdf(sample, mu, sigma2),
	    std::log(0.398942280401433), 1e-3);

	mu = 10.0;
	sigma2 = 100;
	sample = 1.0;
	EXPECT_NEAR(
	    CGaussianDistribution::univariate_log_pdf(sample, mu, sigma2),
	    std::log(0.026608524989875), 1e-3);

	mu = 5.0;
	sigma2 = 25;
	sample = 1.0;
	EXPECT_NEAR(
	    CGaussianDistribution::univariate_log_pdf(sample, mu, sigma2),
	    std::log(0.057938310552297), 1e-3);

	mu = 2.0;
	sigma2 = 16.0;
	sample = 0;
	EXPECT_NEAR(
	    CGaussianDistribution::univariate_log_pdf(sample, mu, sigma2),
	    std::log(0.088016331691075), 1e-3);
}

