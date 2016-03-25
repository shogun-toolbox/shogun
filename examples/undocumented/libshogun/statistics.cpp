/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 * Written (W) 2012 Victor Sadkov
 * Copyright (C) 2011 Moscow State University
 */

#include <shogun/base/init.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

void test_mean()
{
	SGMatrix<float64_t> X(3,5);

	for (index_t i=0; i<X.num_rows*X.num_cols; ++i)
	{
		X.matrix[i]=i;
	}
	X.display_matrix("X");

	SGVector<float64_t> mean=CStatistics::matrix_mean(X, true);
	mean.display_vector("mean");
	ASSERT(mean.vlen==5);
	ASSERT(mean[0]==1);
	ASSERT(mean[1]==4);
	ASSERT(mean[2]==7);
	ASSERT(mean[3]==10);
	ASSERT(mean[4]==13);

	float64_t mean2=linalg::mean(mean);
	ASSERT(mean2==7);

	mean=CStatistics::matrix_mean(X, false);
	mean.display_vector("mean");
	ASSERT(mean.vlen==3);
	ASSERT(mean[0]==6);
	ASSERT(mean[1]==7);
	ASSERT(mean[2]==8);

	mean2=linalg::mean(mean);
	ASSERT(mean2==7);
}

void test_median()
{
	SGMatrix<float64_t> X(3,5);
	SGVector<float64_t> Y(X.num_rows*X.num_cols);
	for (index_t i=0; i<X.num_rows*X.num_cols; ++i)
	{
		X.matrix[i]=CMath::random(0, 15);
		Y[i]=X.matrix[i];
	}
	X.display_matrix("X");
	Y.display_vector("Y");

	/* test all median computation method on vector and matrix */
	float64_t median=CStatistics::median(Y, false, false);
	ASSERT(median==CStatistics::median(Y, false, true));
	ASSERT(median==CStatistics::median(Y, true));

	ASSERT(median==CStatistics::matrix_median(X, false, false));
	ASSERT(median==CStatistics::matrix_median(X, false, true));
	ASSERT(median==CStatistics::matrix_median(X, true));
}

void test_variance()
{
	SGMatrix<float64_t> X(3,5);

	for (index_t i=0; i<X.num_rows*X.num_cols; ++i)
	{
		X.matrix[i]=i;
	}
	X.display_matrix("X");

	SGVector<float64_t> var=CStatistics::matrix_variance(X, true);
	var.display_vector("variance");
	ASSERT(var.vlen==5);
	ASSERT(var[0]==1);
	ASSERT(var[1]==1);
	ASSERT(var[2]==1);
	ASSERT(var[3]==1);
	ASSERT(var[4]==1);

	float64_t var2=CStatistics::variance(var);
	ASSERT(var2==0);

	var=CStatistics::matrix_variance(X, false);
	var.display_vector("variance");
	ASSERT(var.vlen==3);
	ASSERT(var[0]==22.5);
	ASSERT(var[1]==22.5);
	ASSERT(var[2]==22.5);

	var2=CStatistics::variance(var);
	ASSERT(var2==0);
}

void test_confidence_intervals()
{
	int32_t data_size=100;
	SGVector<float64_t> data(data_size);
	data.range_fill();

	float64_t low, up, mean;
	float64_t error_prob=0.1;
	mean=CStatistics::confidence_intervals_mean(data, error_prob, low, up);

	SG_SPRINT("sample mean: %f. True mean lies in [%f,%f] with %f%%\n",
			mean, low, up, 100*(1-error_prob));

	SG_SPRINT("variance: %f\n", CStatistics::variance(data));
	SG_SPRINT("deviation: %f\n", CStatistics::std_deviation(data));
}

void test_inverse_student_t()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::inverse_student_t(1, 0.99);
	SG_SPRINT("inverse_student_t(0.99, 1)=%f\n", difference);
	difference-=31.820515953773953;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-14);

	difference=CStatistics::inverse_student_t(2, 0.99);
	SG_SPRINT("inverse_student_t(0.99, 2)=%f\n", difference);
	difference-= 6.964556734283233;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-14);

	difference=CStatistics::inverse_student_t(3, 0.99);
	SG_SPRINT("inverse_student_t(0.99, 3)=%f\n", difference);
	difference-=4.540702858568132;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-20);

	difference=CStatistics::inverse_student_t(4, 0.99);
	SG_SPRINT("inverse_student_t(0.99, 4)=%f\n", difference);
	difference-=3.746947387979196;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-20);
}

void test_incomplete_gamma()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::incomplete_gamma(2, 1);
	SG_SPRINT("incomplete_gamma(1, 2)=%f\n", difference);
	difference-= 0.264241117657115;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::incomplete_gamma(3, 2);
	SG_SPRINT("incomplete_gamma(3, 2)=%f\n", difference);
	difference-= 0.323323583816937;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::incomplete_gamma(1, 0.1);
	SG_SPRINT("incomplete_gamma(1, 0.1)=%f\n", difference);
	difference-=0.095162581964040;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);
}

void test_gamma_cdf()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::gamma_cdf(0.95, 1, 2);
	SG_SPRINT("gamma_cdf(0.95, 1, 2)=%f\n", difference);
	difference-=0.378114943534980;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::gamma_cdf(0.95, 2, 2);
	SG_SPRINT("gamma_cdf(0.95, 2, 2)=%f\n", difference);
	difference-= 0.082719541714095;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-15);

	difference=CStatistics::gamma_cdf(1, 1, 1);
	SG_SPRINT("gamma_cdf(1, 1, 1)=%f\n", difference);
	difference-= 0.632120558828558;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-15);

	difference=CStatistics::gamma_cdf(0.95, 0.9, 1.1);
	SG_SPRINT("gamma_cdf(0.95, 0.9, 1.1=%f\n", difference);
	difference-= 0.624727614394445;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-15);
}

void test_normal_cdf()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::normal_cdf(1);
	SG_SPRINT("normal_cdf(1)=%f\n", difference);
	difference-=0.841344746068543;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::normal_cdf(2);
	SG_SPRINT("normal_cdf(2)=%f\n", difference);
	difference-=0.977249868051821;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::normal_cdf(0.1);
	SG_SPRINT("normal_cdf(0.1)=%f\n", difference);
	difference-=0.539827837277029;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);
}

void test_inverse_normal_cdf()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::inverse_normal_cdf(0.4, 0, 1);
	SG_SPRINT("inverse_normal_cdf(0.4, 0, 1)=%f\n", difference);
	difference-=-0.253347103135800;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::inverse_normal_cdf(0.8, 0.2, 2.2);
	SG_SPRINT("inverse_normal_cdf(0.8, 0.2, 2.2)=%f\n", difference);
	difference-=2.051566713860412;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::inverse_normal_cdf(0.1, 0.1, 1.2);
	SG_SPRINT("inverse_normal_cdf(0.1, 0.1, 1.2)=%f\n", difference);
	difference-=-1.437861878653521;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);
}

void test_error_function()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::error_function(1);
	SG_SPRINT("error_function(1)=%f\n", difference);
	difference-=0.842700792949715;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::error_function(2);
	SG_SPRINT("error_function(2)=%f\n", difference);
	difference-=0.995322265018953;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::error_function(0.1);
	SG_SPRINT("error_function(0.1)=%f\n", difference);
	difference-=0.112462916018285;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);
}

void test_error_function_complement()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::error_function_complement(1);
	SG_SPRINT("error_function_complement(1)=%f\n", difference);
	difference-=0.157299207050285;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::error_function_complement(2);
	SG_SPRINT("error_function_complement(2)=%f\n", difference);
	difference-=0.004677734981047;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::error_function_complement(0.1);
	SG_SPRINT("error_function_complement(0.1)=%f\n", difference);
	difference-=0.887537083981715;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);
}

void test_inverse_gamma_cdf()
{
	/* some tests for high precision MATLAB comparison */
	float64_t difference=CStatistics::inverse_gamma_cdf(0.5, 1.0, 1.0);
	SG_SPRINT("inverse_gamma_cdf(0.5, 1.0, 1.0)=%f\n", difference);
	difference-=0.693147180559945;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::inverse_gamma_cdf(0.5, 0.5, 0.3);
	SG_SPRINT("inverse_gamma_cdf(0.5, 0.5, 0.3)=%f\n", difference);
	difference-= 0.068240463467936;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);

	difference=CStatistics::inverse_gamma_cdf(0.8, 0.1, 0.3);
	SG_SPRINT("inverse_gamma_cdf(0.8, 0.1, 0.3)=%f\n", difference);
	difference-=0.020816964971992;
	difference=CMath::abs(difference);
	ASSERT(difference<=10E-16);
}

#ifdef HAVE_LAPACK
void test_covariance_matrix()
{
	SGMatrix<float64_t> X(2,3);
	for (index_t i=0; i<X.num_cols*X.num_rows; ++i)
		X.matrix[i]=i;

	X.display_matrix("X");
	SGMatrix<float64_t> cov=CStatistics::covariance_matrix(X);
	cov.display_matrix("cov");

	/* all entries of this covariance matrix will be 0.5 */
	for (index_t i=0; i<cov.num_rows*cov.num_cols; ++i)
		ASSERT(cov.matrix[i]==0.5);

}
#endif //HAVE_LAPACK

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test_mean();
	test_median();
	test_variance();
	test_confidence_intervals();
	test_inverse_student_t();
	test_incomplete_gamma();
	test_gamma_cdf();
	test_inverse_gamma_cdf();
	test_normal_cdf();
	test_inverse_normal_cdf();
	test_error_function();
	test_error_function_complement();

#ifdef HAVE_LAPACK
	test_covariance_matrix();
#endif //HAVE_LAPACK

	exit_shogun();

	return 0;
}

