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
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

void create_fixed_data_kernel_small(CFeatures*& features_p,
		CFeatures*& features_q, CKernel*& kernel_p, CKernel*& kernel_q)
{
	index_t m=2;
	index_t d=3;

	SGMatrix<float64_t> p(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		p.matrix[i]=i;

//	p.display_matrix("p");

	SGMatrix<float64_t> q(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		q.matrix[i]=i+10;

//	q.display_matrix("q");

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

//	p.display_matrix("p");

	SGMatrix<float64_t> q(d,m);
	for (index_t i=0; i<d*m; ++i)
		q.matrix[i]=((i+10)*(i%4+2))%4;

//	q.display_matrix("q");

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
void test_hsic_fixed()
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
	SG_SPRINT("hsic fixed: %f\n", difference);
	ASSERT(CMath::abs(difference-m*0.164761446385339)<10E-16);


	SG_UNREF(hsic);
}

void test_hsic_gamma()
{
	CFeatures* features_p=NULL;
	CFeatures* features_q=NULL;
	CKernel* kernel_p=NULL;
	CKernel* kernel_q=NULL;
	create_fixed_data_kernel_big(features_p, features_q, kernel_p, kernel_q);

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	hsic->set_null_approximation_method(HSIC_GAMMA);
	float64_t p=hsic->compute_p_value(0.05);
	SG_SPRINT("p-value: %f\n", p);
	ASSERT(CMath::abs(p-0.172182287884256)<10E-15);

	SG_UNREF(hsic);
}

void test_hsic_sample_null()
{
	CFeatures* features_p=NULL;
	CFeatures* features_q=NULL;
	CKernel* kernel_p=NULL;
	CKernel* kernel_q=NULL;
	create_fixed_data_kernel_big(features_p, features_q, kernel_p, kernel_q);

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	/* do sampling null */
	hsic->set_null_approximation_method(PERMUTATION);
	float64_t p=hsic->compute_p_value(0.05);
	SG_SPRINT("p-value: %f\n", p);

	/* ensure that sampling null of hsic leads to same results as using
	 * CKernelIndependenceTest */
	CMath::init_random(1);
	float64_t mean1=linalg::mean(hsic->sample_null());
	float64_t var1=CStatistics::variance(hsic->sample_null());
	SG_SPRINT("mean1=%f, var1=%f\n", mean1, var1);

	CMath::init_random(1);
	float64_t mean2=linalg::mean(
			hsic->CKernelIndependenceTest::sample_null());
	float64_t var2=CStatistics::variance(hsic->sample_null());
	SG_SPRINT("mean2=%f, var2=%f\n", mean2, var2);

	/* assert than results are the same from bot sampling null impl. */
	ASSERT(CMath::abs(mean1-mean2)<10E-8);
	ASSERT(CMath::abs(var1-var2)<10E-8);

	SG_UNREF(hsic);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test_hsic_fixed();
	test_hsic_gamma();
	test_hsic_sample_null();

	exit_shogun();
	return 0;
}

