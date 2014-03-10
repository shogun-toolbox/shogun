/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 pl8787
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
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
TEST(HSICTEST, hsic_fixed)
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
	//SG_SPRINT("hsic fixed: %f\n", difference);
	EXPECT_NEAR(difference, m*0.164761446385339, 1e-15);

	SG_UNREF(hsic);
}

TEST(HSICTEST, hsic_gamma)
{
	CFeatures* features_p=NULL;
	CFeatures* features_q=NULL;
	CKernel* kernel_p=NULL;
	CKernel* kernel_q=NULL;
	create_fixed_data_kernel_big(features_p, features_q, kernel_p, kernel_q);

	CHSIC* hsic=new CHSIC(kernel_p, kernel_q, features_p, features_q);

	hsic->set_null_approximation_method(HSIC_GAMMA);
	float64_t p=hsic->compute_p_value(0.05);
	//SG_SPRINT("p-value: %f\n", p);
	EXPECT_NEAR(p, 0.172182287884256, 1e-14);

	SG_UNREF(hsic);
}

TEST(HSICTEST, hsic_sample_null)
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
	//SG_SPRINT("p-value: %f\n", p);
	EXPECT_NEAR(p, 0.576000, 1e-14);

	/* ensure that sampling null of hsic leads to same results as using
	 * CKernelIndependenceTest */
	CMath::init_random(1);
	float64_t mean1=CStatistics::mean(hsic->sample_null());
	float64_t var1=CStatistics::variance(hsic->sample_null());
	//SG_SPRINT("mean1=%f, var1=%f\n", mean1, var1);

	CMath::init_random(1);
	float64_t mean2=CStatistics::mean(
			hsic->CKernelIndependenceTest::sample_null());
	float64_t var2=CStatistics::variance(hsic->sample_null());
	//SG_SPRINT("mean2=%f, var2=%f\n", mean2, var2);

	/* assert than results are the same from bot sampling null impl. */
	EXPECT_NEAR(mean1, mean2, 1e-7);
	EXPECT_NEAR(var1, var2, 1e-7);

	SG_UNREF(hsic);
}

