/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/HSIC.h>
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

/** tests the hsic statistic for a single fixed data case and ensures
 * equality with matlab implementation */
void test_hsic_fixed()
{
	index_t m=2;
	index_t d=3;
	float64_t sigma_x=2;
	float64_t sq_sigma_x_twice=sigma_x*sigma_x*2;
	float64_t sigma_y=3;
	float64_t sq_sigma_y_twice=sigma_y*sigma_y*2;

	SGMatrix<float64_t> p(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		p.matrix[i]=i;

//	p.display_matrix("p");

	SGMatrix<float64_t> q(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		q.matrix[i]=i+10;

//	q.display_matrix("q");

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(q);

	/* shoguns kernel width is different */
	CGaussianKernel* kernel_p=new CGaussianKernel(10, sq_sigma_x_twice);
	CGaussianKernel* kernel_q=new CGaussianKernel(10, sq_sigma_y_twice);

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	/* assert matlab result */
	float64_t difference=hsic->compute_statistic();
	SG_SPRINT("hsic fixed: %f\n", difference);
	ASSERT(CMath::abs(difference-0.164761446385339)<10E-17);

	SG_UNREF(hsic);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	/* all tests have been "speed up" by reducing the number of runs/samples.
	 * If you have any doubts in the results, set all num_runs to original
	 * numbers and activate asserts. If they fail, something is wrong.
	 */
	test_hsic_fixed();

	exit_shogun();
	return 0;
}

