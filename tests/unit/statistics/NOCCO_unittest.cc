/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * Written (w) 2012-2013 Heiko Strathmann
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
 */

#include <shogun/statistics/NOCCO.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/eigen3.h>
#include <gtest/gtest.h>

using namespace shogun;

using namespace Eigen;

/** tests the nocco statistic for a single fixed data case and ensures
 * equality with matlab implementation */
TEST(NOCCO, compute_statistic)
{
	const index_t m=2;
	const index_t d=3;
	const float64_t epsilon=0.1;

	SGMatrix<float64_t> p(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		p.matrix[i]=i;

	SGMatrix<float64_t> q(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		q.matrix[i]=i+10;

	CFeatures* features_p=new CDenseFeatures<float64_t>(p);
	CFeatures* features_q=new CDenseFeatures<float64_t>(q);

	float64_t sigma_x=2;
	float64_t sigma_y=3;
	float64_t sq_sigma_x_twice=sigma_x*sigma_x*2;
	float64_t sq_sigma_y_twice=sigma_y*sigma_y*2;

	/* shoguns kernel width is different */
	CKernel* kernel_p=new CGaussianKernel(10, sq_sigma_x_twice);
	CKernel* kernel_q=new CGaussianKernel(10, sq_sigma_y_twice);

	CNOCCO* nocco=new CNOCCO(kernel_p, kernel_q, features_p, features_q);
	nocco->set_epsilon(epsilon);

	float64_t statistic=nocco->compute_statistic();

	/* compute the statistic locally */
	kernel_p->init(features_p, features_p);
	kernel_q->init(features_q, features_q);

	SGMatrix<float64_t> K=kernel_p->get_kernel_matrix();
	SGMatrix<float64_t> L=kernel_q->get_kernel_matrix();

	K.center();
	L.center();

	Map<MatrixXd> Km(K.matrix, K.num_rows, K.num_cols);
	Map<MatrixXd> Lm(L.matrix, L.num_rows, L.num_cols);

	const MatrixXd& Km_inv=(Km+2*m*epsilon*MatrixXd::Identity(2*m, 2*m)).inverse();
	const MatrixXd& Lm_inv=(Lm+2*m*epsilon*MatrixXd::Identity(2*m, 2*m)).inverse();

	float64_t naive=(Km*Km_inv*Lm*Lm_inv).trace();

	/* assert locally computed naive result */
	EXPECT_NEAR(statistic, naive, 1E-15);

	SG_UNREF(nocco);
}

TEST(NOCCO, compute_p_value)
{
	const index_t m=2;
	const index_t d=3;
	const float64_t epsilon=0.1;

	SGMatrix<float64_t> p(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		p.matrix[i]=i;

	SGMatrix<float64_t> q(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		q.matrix[i]=i+10;

	CFeatures* features_p=new CDenseFeatures<float64_t>(p);
	CFeatures* features_q=new CDenseFeatures<float64_t>(q);

	float64_t sigma_x=2;
	float64_t sigma_y=3;
	float64_t sq_sigma_x_twice=sigma_x*sigma_x*2;
	float64_t sq_sigma_y_twice=sigma_y*sigma_y*2;

	/* shoguns kernel width is different */
	CKernel* kernel_p=new CGaussianKernel(10, sq_sigma_x_twice);
	CKernel* kernel_q=new CGaussianKernel(10, sq_sigma_y_twice);

	CNOCCO* nocco=new CNOCCO(kernel_p, kernel_q, features_p, features_q);
	nocco->set_epsilon(epsilon);

	/* compute p-value via sampling null */
	nocco->set_null_approximation_method(PERMUTATION);
	EXPECT_NEAR(nocco->compute_p_value(0.05), 1.0, 1E-15);

	SG_UNREF(nocco);
}

TEST(NOCCO, sample_null)
{
	const index_t m=10;
	const index_t d=7;
	const float64_t epsilon=0.1;

	SGMatrix<float64_t> p(d,m);
	for (index_t i=0; i<d*m; ++i)
		p.matrix[i]=(i+8)%3;

	SGMatrix<float64_t> q(d,m);
	for (index_t i=0; i<d*m; ++i)
		q.matrix[i]=((i+10)*(i%4+2))%4;

	CFeatures* features_p=new CDenseFeatures<float64_t>(p);
	CFeatures* features_q=new CDenseFeatures<float64_t>(q);

	float64_t sigma_x=2;
	float64_t sigma_y=3;
	float64_t sq_sigma_x_twice=sigma_x*sigma_x*2;
	float64_t sq_sigma_y_twice=sigma_y*sigma_y*2;

	/* shogun's kernel width is different */
	CKernel* kernel_p=new CGaussianKernel(10, sq_sigma_x_twice);
	CKernel* kernel_q=new CGaussianKernel(10, sq_sigma_y_twice);

	CNOCCO* nocco=new CNOCCO(kernel_p, kernel_q, features_p, features_q);
	nocco->set_epsilon(epsilon);

	/* do sampling null */

	/* ensure that sampling null of nocco leads to same results as using
	 * CKernelIndependenceTest */
	CMath::init_random(1);
	float64_t mean1=CStatistics::linalg(nocco->sample_null());
	float64_t var1=CStatistics::variance(nocco->sample_null());

	CMath::init_random(1);
	float64_t mean2=CStatistics::linalg(
			nocco->CKernelIndependenceTest::sample_null());
	float64_t var2=CStatistics::variance(nocco->sample_null());

	/* assert than results are the same from bot sampling null impl. */
	EXPECT_NEAR(mean1, mean2, 1E-8);
	EXPECT_NEAR(var1, var2, 1E-8);

	SG_UNREF(nocco);
}
