/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016-2017 Heiko Strathmann, Dougal Sutherland
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

#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/kernel_exp_family/impl/Full.h>
#include <shogun/distributions/kernel_exp_family/impl/kernel/Gaussian.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/some.h>
#include <shogun/base/init.h>
#include <gtest/gtest.h>
#include <memory>

#include "DataFixture.h"

using namespace std;
using namespace shogun;
using namespace kernel_exp_family_impl;

/* All unit tests are based on the following gist
 * https://gist.github.com/karlnapf/c0b24fc18d946cc315733ed679e249e8
 */
class KernelExpFamilyImplFullFixed: public DataFixture, public ::testing::Test
{
public:
	void SetUp()
	{
		DataFixture::SetUp();
		auto sigma = 2.0;
		auto lambda = 1.0;

		auto X = get_data_train();
		auto kernel = make_shared<kernel::Gaussian>(sigma);
		est = make_shared<Full>(X, kernel, lambda);
	}

	virtual SGMatrix<float64_t> get_data_train() { return X_train_fixed; }

	 shared_ptr<Full> est;
};

class KernelExpFamilyImplFullRandom: public KernelExpFamilyImplFullFixed
{
	virtual SGMatrix<float64_t> get_data_train() { return X_train_random; }
};

TEST_F(KernelExpFamilyImplFullFixed, compute_h_kernel_Gaussian)
{
	auto result = est->compute_h();

	float64_t reference[] = {0.00902188,  0.01353302,  0.01834103,  0.04119238,
							 -0.02736291,-0.0547254 };
	ASSERT_EQ(result.vlen, ND);
	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-8);
}

TEST_F(KernelExpFamilyImplFullFixed, fit_kernel_Gaussian)
{
	est->fit();
	auto result = est->get_beta();
	ASSERT_EQ(result.vlen, ND);
	ASSERT_TRUE(result.vector);
	
	float64_t reference[] = {
			0.0022809128878491,  0.0034202348297001,  0.0040642480460088,
			0.0092514041335136, -0.006461025579696 , -0.0129449913732097
	};

	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-8);
	
}

TEST_F(KernelExpFamilyImplFullFixed, log_pdf_kernel_Gaussian)
{
	est->fit();
	est->set_data(X_test_fixed);

	auto result = est->log_pdf();

	float64_t reference[] = {
			0.6610996707107812, 0.1853735804373066
	};

	ASSERT_TRUE(result.vector);
	ASSERT_EQ(result.vlen, N_test);
	for (auto i=0; i<N_test; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-8);
}

TEST_F(KernelExpFamilyImplFullFixed, grad_kernel_Gaussian)
{
	est->fit();
	est->set_data(X_test_fixed);
	auto grad = est->grad(0);

	float64_t reference[] = {-0.00684274, -0.0102607 };
	ASSERT_EQ(grad.vlen, D);
	ASSERT_TRUE(grad.vector);
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference[i], 1e-8);
	
	grad = est->grad(1);
	float64_t reference2[] = {-0.62020189, -0.03895487};
	ASSERT_EQ(grad.vlen, D);
	ASSERT_TRUE(grad.vector);
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference2[i], 1e-8);
}

TEST_F(KernelExpFamilyImplFullFixed, hessian_kernel_Gaussian)
{
	est->fit();
	est->set_data(X_test_fixed);
	auto hessian = est->hessian(0);
	
	float64_t reference[] = {-1.34299555, -0.02133143,
							-0.02133143, -1.36075253};
	ASSERT_EQ(hessian.num_rows, D);
	ASSERT_EQ(hessian.num_cols, D);
	ASSERT_TRUE(hessian.matrix);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference[i], 1e-8);

	hessian = est->hessian(1);

	float64_t reference2[] = { 0.40612246, -0.02956325,
								-0.02956325, -0.67672628};
	ASSERT_EQ(hessian.num_rows, D);
	ASSERT_EQ(hessian.num_cols, D);
	ASSERT_TRUE(hessian.matrix);
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference2[i], 1e-8);
}

TEST_F(KernelExpFamilyImplFullRandom, hessian_diag_equals_hessian)
{
	est->fit();
	for (auto i=0; i<est->get_num_data(); i++)
	{
		auto hessian = est->hessian(i);
		auto hessian_diag = est->hessian_diag(i);

		ASSERT_EQ(hessian.num_rows, D);
		ASSERT_EQ(hessian.num_cols, D);
		ASSERT_TRUE(hessian.matrix);

		ASSERT_EQ(hessian_diag.vlen, D);
		ASSERT_TRUE(hessian_diag.vector);

		for (auto j=0; j<D; j++)
			EXPECT_NEAR(hessian_diag[j], hessian(j,j), 1e-8);
	}
}

TEST_F(KernelExpFamilyImplFullFixed, score_kernel_Gaussian)
{
	est->fit();
	EXPECT_NEAR(est->score(), -2.56147602838, 1e-8);

	est->set_data(X_test_fixed);
	EXPECT_NEAR(est->score(), -1.39059595876, 1e-8);
}
