/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Heiko Strathmann, pl8787
 */

#include <shogun/base/init.h>
#include <shogun/statistics/HSIC.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <gtest/gtest.h>

using namespace shogun;

void create_fixed_data_kernel_small(CFeatures*& features_p,
		CFeatures*& features_q, CKernel*& kernel_p, CKernel*& kernel_q)
{
	index_t m=2;
	index_t d=3;

	SGMatrix<float64_t> p(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		p.matrix[i]=i;

	SGMatrix<float64_t> q(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		q.matrix[i]=i+10;

	features_p=new CDenseFeatures<float64_t>(p);
	features_q=new CDenseFeatures<float64_t>(q);

	float64_t sigma_x=2;
	float64_t sigma_y=3;
	float64_t sq_sigma_x_twice=sigma_x*sigma_x*2;
	float64_t sq_sigma_y_twice=sigma_y*sigma_y*2;

	/* shoguns kernel width is different */
	kernel_p=new CGaussianKernel(10, sq_sigma_x_twice);
	kernel_q=new CGaussianKernel(10, sq_sigma_y_twice);
}

void create_fixed_data_kernel_big(CFeatures*& features_p,
		CFeatures*& features_q, CKernel*& kernel_p, CKernel*& kernel_q)
{
	index_t m=10;
	index_t d=7;

	SGMatrix<float64_t> p(d,m);
	for (index_t i=0; i<d*m; ++i)
		p.matrix[i]=(i+8)%3;

	SGMatrix<float64_t> q(d,m);
	for (index_t i=0; i<d*m; ++i)
		q.matrix[i]=((i+10)*(i%4+2))%4;

	features_p=new CDenseFeatures<float64_t>(p);
	features_q=new CDenseFeatures<float64_t>(q);

	float64_t sigma_x=2;
	float64_t sigma_y=3;
	float64_t sq_sigma_x_twice=sigma_x*sigma_x*2;
	float64_t sq_sigma_y_twice=sigma_y*sigma_y*2;

	/* shoguns kernel width is different */
	kernel_p=new CGaussianKernel(10, sq_sigma_x_twice);
	kernel_q=new CGaussianKernel(10, sq_sigma_y_twice);
}

/** tests the hsic statistic for a single fixed data case and ensures
 * equality with sma implementation */
TEST(HSIC, hsic_fixed)
{
	CFeatures* features_p=NULL;
	CFeatures* features_q=NULL;
	CKernel* kernel_p=NULL;
	CKernel* kernel_q=NULL;
	create_fixed_data_kernel_small(features_p, features_q, kernel_p, kernel_q);

	index_t m=features_p->get_num_vectors();

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	/* assert matlab result, note that compute statistic computes m*hsic */
	float64_t difference=hsic->compute_statistic();

	EXPECT_NEAR(difference, m*0.164761446385339, 1e-15);

	SG_UNREF(hsic);
}

TEST(HSIC, hsic_gamma)
{
	CFeatures* features_p=NULL;
	CFeatures* features_q=NULL;
	CKernel* kernel_p=NULL;
	CKernel* kernel_q=NULL;
	create_fixed_data_kernel_big(features_p, features_q, kernel_p, kernel_q);

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	hsic->set_null_approximation_method(HSIC_GAMMA);
	float64_t p=hsic->compute_p_value(0.05);

	EXPECT_NEAR(p, 0.172182287884256, 1e-14);

	SG_UNREF(hsic);
}

TEST(HSIC, hsic_sample_null)
{
	CFeatures* features_p=NULL;
	CFeatures* features_q=NULL;
	CKernel* kernel_p=NULL;
	CKernel* kernel_q=NULL;
	create_fixed_data_kernel_big(features_p, features_q, kernel_p, kernel_q);

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	/* do sampling null */
	hsic->set_null_approximation_method(PERMUTATION);
	hsic->compute_p_value(0.05);

	/* ensure that sampling null of hsic leads to same results as using
	 * CKernelIndependenceTest */
	CMath::init_random(1);
	float64_t mean1=CStatistics::mean(hsic->sample_null());
	float64_t var1=CStatistics::variance(hsic->sample_null());

	CMath::init_random(1);
	float64_t mean2=CStatistics::mean(
			hsic->CKernelIndependenceTest::sample_null());
	float64_t var2=CStatistics::variance(hsic->sample_null());

	/* assert than results are the same from bot sampling null impl. */
	EXPECT_NEAR(mean1, mean2, 1e-7);
	EXPECT_NEAR(var1, var2, 1e-7);

	SG_UNREF(hsic);
}

